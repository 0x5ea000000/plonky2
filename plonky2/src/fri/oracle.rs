#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
use core::alloc::Layout;
use core::mem::transmute;
use core::ptr::NonNull;
use std::alloc::{AllocError, Allocator};
use std::sync::Arc;

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
use rustacuda::memory::{AsyncCopyDestination, DeviceBuffer, DeviceSlice};
use rustacuda::prelude::Context;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::{fri_proof, fri_proof_with_gpu};
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::{MerkleCap, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::{self, TimingTree};
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

#[derive(Debug)]
pub struct CudaInnerContext {
    pub stream: rustacuda::stream::Stream,
    pub stream2: rustacuda::stream::Stream,
}

#[derive(Debug)]
pub struct CudaAllocator {}

unsafe impl Allocator for CudaAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let raw_ptr = rustacuda::memory::cuda_malloc_locked::<u8>(layout.size()).unwrap();
            let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: `layout` is non-zero in size,
            // other conditions must be upheld by the caller
            unsafe {
                // dealloc(ptr.as_ptr(), layout)
                rustacuda::memory::cuda_free_locked(ptr.as_ptr()).unwrap();
            }
        }
    }
}

#[repr(C)]
#[derive(Debug)]
// pub struct CudaInvContext<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
pub struct CudaInvContext<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub inner: CudaInnerContext,
    pub ext_values_flatten: Arc<Vec<F>>,
    pub values_flatten: Arc<Vec<F, CudaAllocator>>,
    pub digests_and_caps_buf: Arc<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>,

    pub ext_values_flatten2: Arc<Vec<F>>,
    pub values_flatten2: Arc<Vec<F, CudaAllocator>>,
    pub digests_and_caps_buf2: Arc<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>,

    pub ext_values_flatten3: Arc<Vec<F>>,
    pub values_flatten3: Arc<Vec<F, CudaAllocator>>,
    pub digests_and_caps_buf3: Arc<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>,

    // pub values_device: DeviceBuffer::<F>,
    // pub ext_values_device: DeviceBuffer::<F>,
    pub cache_mem_device: DeviceBuffer<F>,
    pub second_stage_offset: usize,

    pub root_table_device: DeviceBuffer<F>,
    pub root_table_device2: DeviceBuffer<F>,
    pub constants_sigmas_commitment_leaves_device: DeviceBuffer<F>,
    pub shift_powers_device: DeviceBuffer<F>,
    pub shift_inv_powers_device: DeviceBuffer<F>,

    pub points_device: DeviceBuffer<F>,
    pub z_h_on_coset_evals_device: DeviceBuffer<F>,
    pub z_h_on_coset_inverses_device: DeviceBuffer<F>,
    pub k_is_device: DeviceBuffer<F>,

    pub ctx: Context,
}

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.leaves[index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        );

        fri_proof
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    pub fn from_values_with_gpu(
        values: &Vec<F>,
        poly_num: usize,
        values_num_per_poly: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
        ctx: &mut CudaInvContext<F, C, D>,
    ) -> Self {
        // let poly_num: usize = values.len();
        // let values_num_per_poly  = values[0].values.len();

        // let values_flatten = values;
        // let mut values_flatten = timed!(
        //     timing,
        //     "flat map",
        //     values.into_par_iter().flat_map(|poly| poly.values).collect::<Vec<F>>()
        // );

        let salt_size = if blinding { SALT_SIZE } else { 0 };

        let lg_n = log2_strict(values_num_per_poly);
        let n_inv = F::inverse_2exp(lg_n);
        let n_inv_ptr: *const F = &n_inv;

        let len_cap = (1 << cap_height);
        let num_digests = 2 * (values_num_per_poly * (1 << rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;

        let values_flatten_len = poly_num * values_num_per_poly;
        let ext_values_flatten_len =
            (values_flatten_len + salt_size * values_num_per_poly) * (1 << rate_bits);
        let digests_and_caps_buf_len = num_digests_and_caps;

        let pad_extvalues_len = ext_values_flatten_len;

        let (ext_values_flatten, values_flatten, digests_and_caps_buf);

        let is_first_stage;
        let ext_values_device_offset;
        if values_num_per_poly * poly_num == ctx.values_flatten.len() {
            println!("in first stage");
            ext_values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.ext_values_flatten).unwrap();
            values_flatten =
                Arc::<Vec<F, CudaAllocator>>::get_mut(&mut ctx.values_flatten).unwrap();
            digests_and_caps_buf =
                Arc::<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>::get_mut(
                    &mut ctx.digests_and_caps_buf,
                )
                .unwrap();
            ext_values_device_offset = 0;
            is_first_stage = true;
        } else {
            // } else if values_num_per_poly*poly_num == ctx.values_flatten2.len() {
            println!("in second stage");
            ext_values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.ext_values_flatten2).unwrap();
            values_flatten =
                Arc::<Vec<F, CudaAllocator>>::get_mut(&mut ctx.values_flatten2).unwrap();
            digests_and_caps_buf =
                Arc::<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>::get_mut(
                    &mut ctx.digests_and_caps_buf2,
                )
                .unwrap();
            ext_values_device_offset = ctx.second_stage_offset;
            is_first_stage = false;
        }

        // let (values_device, ext_values_device) =
        //     ctx.cache_mem_device.split_at_mut(ext_values_device_offset).1.split_at_mut(values_flatten_len);
        let values_device = ctx
            .cache_mem_device
            .split_at_mut(ext_values_device_offset)
            .1;

        // let values_device = &mut ctx.cache_mem_device[0..values_flatten_len];
        // let ext_values_device = &mut ctx.cache_mem_device[values_flatten_len..];
        let root_table_device = &ctx.root_table_device;
        let root_table_device2 = &ctx.root_table_device2;
        let shift_powers_device = &ctx.shift_powers_device;

        timed!(timing, "copy values", unsafe {
            transmute::<&mut DeviceSlice<F>, &mut DeviceSlice<u64>>(
                &mut values_device[0..values_flatten_len],
            )
            .async_copy_from(transmute::<&Vec<F>, &Vec<u64>>(values), &ctx.inner.stream)
            .unwrap();
            ctx.inner.stream.synchronize().unwrap();
        });
        // unsafe {
        //     let ctx_ptr :*mut CudaInnerContext = &mut ctx.inner;
        //     timed!(
        //         timing,
        //         "FFT + build Merkle tree + transpose with gpu",
        //         {
        //             plonky2_cuda::merkle_tree_from_values(
        //                 values_device.as_mut_ptr() as *mut u64,
        //                 ext_values_device.as_mut_ptr() as *mut u64,
        //                 poly_num as i32, values_num_per_poly as i32,
        //                 lg_n as i32,
        //                 root_table_device.as_ptr() as *const u64,
        //                 root_table_device2.as_ptr() as *const u64,
        //                 shift_powers_device.as_ptr() as *const u64,
        //                 n_inv_ptr as *const u64,
        //                 rate_bits as i32,
        //                 salt_size as i32,
        //                 cap_height as i32,
        //                 pad_extvalues_len as i32,
        //                 ctx_ptr as *mut core::ffi::c_void,
        //             );
        //         }
        //     );
        // }

        unsafe {
            let ctx_ptr: *mut CudaInnerContext = &mut ctx.inner;
            timed!(timing, "FFT + build Merkle tree + transpose with gpu", {
                plonky2_cuda::ifft(
                    values_device.as_mut_ptr() as *mut u64,
                    poly_num as i32,
                    values_num_per_poly as i32,
                    lg_n as i32,
                    root_table_device.as_ptr() as *const u64,
                    n_inv_ptr as *const u64,
                    ctx_ptr as *mut core::ffi::c_void,
                );

                unsafe {
                    transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(
                        &values_device[0..values_flatten_len],
                    )
                    .async_copy_to(
                        transmute::<&mut Vec<F, CudaAllocator>, &mut Vec<u64>>(values_flatten),
                        &ctx.inner.stream2,
                    )
                    .unwrap();
                }

                plonky2_cuda::merkle_tree_from_coeffs(
                    values_device.as_mut_ptr() as *mut u64,
                    values_device.as_mut_ptr() as *mut u64,
                    poly_num as i32,
                    values_num_per_poly as i32,
                    lg_n as i32,
                    root_table_device.as_ptr() as *const u64,
                    root_table_device2.as_ptr() as *const u64,
                    shift_powers_device.as_ptr() as *const u64,
                    rate_bits as i32,
                    salt_size as i32,
                    cap_height as i32,
                    pad_extvalues_len as i32,
                    ctx_ptr as *mut core::ffi::c_void,
                );
            });
        }
        timed!(timing, "copy result", {
            let mut alllen = ext_values_flatten_len;
            assert!(ext_values_flatten.len() == ext_values_flatten_len);

            // if is_first_stage
            // {
            //     unsafe {
            //         transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(&values_device[0..alllen]).async_copy_to(
            //         transmute::<&mut Vec<F>, &mut Vec<u64>>(ext_values_flatten),
            //         &ctx.inner.stream).unwrap();
            //         ctx.inner.stream.synchronize().unwrap();
            //     }
            // }

            alllen += pad_extvalues_len;

            let len_with_F = digests_and_caps_buf_len * 4;
            let fs = unsafe { transmute::<&mut Vec<_>, &mut Vec<F>>(digests_and_caps_buf) };

            unsafe {
                fs.set_len(len_with_F);
            }
            println!(
                "alllen: {}, digest_and_cap_buf_len: {}, diglen: {}",
                alllen, len_with_F, digests_and_caps_buf_len
            );
            unsafe {
                transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(
                    &values_device[alllen..alllen + len_with_F],
                )
                .async_copy_to(
                    transmute::<&mut Vec<F>, &mut Vec<u64>>(fs),
                    &ctx.inner.stream,
                )
                .unwrap();
                ctx.inner.stream.synchronize().unwrap();
            }

            unsafe {
                fs.set_len(len_with_F / 4);
            }
        });

        let coeffs = values_flatten
            .par_chunks(values_num_per_poly)
            .map(|chunk| PolynomialCoeffs {
                coeffs: chunk.to_vec(),
            })
            .collect::<Vec<_>>();

        // let lde_values = ext_values_flatten
        //     .chunks(values_num_per_poly * (1 << rate_bits)).map(|chunk| chunk.to_vec()).collect::<Vec<_>>();

        // let leaves = timed!(timing, "build leaves",
        //     ext_values_flatten.par_chunks(poly_num+salt_size).map(|chunk| chunk.to_vec()).collect::<Vec<_>>());

        {
            let polynomials = coeffs;

            // let leaves = lde_values;
            // let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
            // timed!(timing, "reverse index bits", reverse_index_bits_in_place(&mut leaves));

            // let merkle_tree = timed!(
            //     timing,
            //     "build Merkle tree",
            //     MerkleTree::new2(leaves, cap_height)
            // );

            let (ctx_ext_values_flatten, ctx_digests_and_caps_buf);
            if is_first_stage {
                ctx_ext_values_flatten = ctx.ext_values_flatten.clone();
                ctx_digests_and_caps_buf = ctx.digests_and_caps_buf.clone();
            } else {
                // } else if values_num_per_poly*poly_num == ctx.values_flatten2.len() {
                ctx_ext_values_flatten = ctx.ext_values_flatten2.clone();
                ctx_digests_and_caps_buf = ctx.digests_and_caps_buf2.clone();
            }

            let mut my_leaves_dev_offset = 0;
            // if !is_first_stage {
            my_leaves_dev_offset = ext_values_device_offset as isize;
            // }
            let ctx_ext_values_flatten_len = ctx_ext_values_flatten.len();
            let merkle_tree = MerkleTree {
                leaves: vec![],
                // leaves,
                // digests: digests_and_caps_buf[0..num_digests].to_vec(),
                digests: vec![],
                cap: MerkleCap(
                    ctx_digests_and_caps_buf[num_digests..num_digests_and_caps].to_vec(),
                ),
                my_leaf_len: poly_num + salt_size,
                my_leaves: ctx_ext_values_flatten,
                my_leaves_len: ctx_ext_values_flatten_len,
                my_leaves_dev_offset,
                my_digests: ctx_digests_and_caps_buf,
            };

            // for (idx, h) in merkle_tree.digests.iter().enumerate() {
            //     // println!("hash: {:?}",  unsafe{std::mem::transmute::<&_, &[u8;32]>(&res)})
            //     let hex_string: String = unsafe{std::mem::transmute::<&_, &[u8;32]>(h)}.iter().map(|byte| format!("{:02x}", byte)).collect();
            //     let result: String = hex_string.chars()
            //         .collect::<Vec<char>>()
            //         .chunks(16)
            //         .map(|chunk| chunk.iter().collect::<String>())
            //         .collect::<Vec<String>>()
            //         .join(", ");
            //     println!("idx: {}, hash: {}", idx, result);
            // }
            //
            // for (idx, h) in merkle_tree.cap.0.iter().enumerate() {
            //     // println!("hash: {:?}",  unsafe{std::mem::transmute::<&_, &[u8;32]>(&res)})
            //     let hex_string: String = unsafe{std::mem::transmute::<&_, &[u8;32]>(h)}.iter().map(|byte| format!("{:02x}", byte)).collect();
            //     let result: String = hex_string.chars()
            //         .collect::<Vec<char>>()
            //         .chunks(16)
            //         .map(|chunk| chunk.iter().collect::<String>())
            //         .collect::<Vec<String>>()
            //         .join(", ");
            //     println!("cap idx: {}, hash: {}", idx, result);
            // }

            Self {
                polynomials,
                merkle_tree,
                degree_log: lg_n,
                rate_bits,
                blinding,
            }
        }
    }

    pub fn from_coeffs_with_gpu(
        quotient_polys_offset: usize,
        values_num_per_poly: usize,
        poly_num: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        ctx: &mut CudaInvContext<F, C, D>,
    ) -> Self {
        // let poly_num: usize = values.len();
        // let values_num_per_poly  = values[0].values.len();

        // let values_flatten = values;
        // let mut values_flatten = timed!(
        //     timing,
        //     "flat map",
        //     values.into_par_iter().flat_map(|poly| poly.values).collect::<Vec<F>>()
        // );

        let salt_size = if blinding { SALT_SIZE } else { 0 };

        let lg_n = log2_strict(values_num_per_poly);
        let n_inv = F::inverse_2exp(lg_n);
        let n_inv_ptr: *const F = &n_inv;

        let len_cap = (1 << cap_height);
        let num_digests = 2 * (values_num_per_poly * (1 << rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;

        let values_flatten_len = poly_num * values_num_per_poly;
        let ext_values_flatten_len =
            (values_flatten_len + salt_size * values_num_per_poly) * (1 << rate_bits);
        let digests_and_caps_buf_len = num_digests_and_caps;

        let pad_extvalues_len = ext_values_flatten_len;

        let values_flatten =
            Arc::<Vec<F, CudaAllocator>>::get_mut(&mut ctx.values_flatten3).unwrap();
        let ext_values_flatten = Arc::<Vec<F>>::get_mut(&mut ctx.ext_values_flatten3).unwrap();
        let digests_and_caps_buf =
            Arc::<Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash>>::get_mut(
                &mut ctx.digests_and_caps_buf3,
            )
            .unwrap();

        println!(
            "hello asdf, cache: {}, quotient offset: {}, values: {}",
            ctx.cache_mem_device.len(),
            quotient_polys_offset,
            values_flatten_len
        );
        let values_device = ctx.cache_mem_device.split_at_mut(quotient_polys_offset).1;
        println!("bye   asdf");

        // let values_device = &mut ctx.cache_mem_device[0..values_flatten_len];
        // let ext_values_device = &mut ctx.cache_mem_device[values_flatten_len..];
        let root_table_device = &ctx.root_table_device;
        let root_table_device2 = &ctx.root_table_device2;
        let shift_powers_device = &ctx.shift_powers_device;

        unsafe {
            let ctx_ptr: *mut CudaInnerContext = &mut ctx.inner;
            timed!(timing, "FFT + build Merkle tree + transpose with gpu", {
                unsafe {
                    transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(
                        &values_device[0..values_flatten_len],
                    )
                    .async_copy_to(
                        transmute::<&mut Vec<F, CudaAllocator>, &mut Vec<u64>>(values_flatten),
                        &ctx.inner.stream2,
                    )
                    .unwrap();
                }

                plonky2_cuda::merkle_tree_from_coeffs(
                    values_device.as_mut_ptr() as *mut u64,
                    values_device.as_mut_ptr() as *mut u64,
                    poly_num as i32,
                    values_num_per_poly as i32,
                    lg_n as i32,
                    root_table_device.as_ptr() as *const u64,
                    root_table_device2.as_ptr() as *const u64,
                    shift_powers_device.as_ptr() as *const u64,
                    rate_bits as i32,
                    salt_size as i32,
                    cap_height as i32,
                    pad_extvalues_len as i32,
                    ctx_ptr as *mut core::ffi::c_void,
                );
            });
        }
        timed!(timing, "copy result", {
            // let alllen = values_flatten_len;
            //
            // unsafe {
            //     transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(&values_device[0..values_flatten_len]).async_copy_to(
            //     transmute::<&mut Vec<F>, &mut Vec<u64>>(values_flatten),
            //     &ctx.inner.stream).unwrap();
            //     ctx.inner.stream.synchronize().unwrap();
            // }

            let mut alllen = ext_values_flatten_len;
            assert!(ext_values_flatten.len() == ext_values_flatten_len);

            // if isFirstStage
            // {
            //     unsafe {
            //         transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(&values_device[0..alllen]).async_copy_to(
            //         transmute::<&mut Vec<F>, &mut Vec<u64>>(ext_values_flatten),
            //         &ctx.inner.stream).unwrap();
            //         ctx.inner.stream.synchronize().unwrap();
            //     }
            // }

            alllen += pad_extvalues_len;

            let len_with_F = digests_and_caps_buf_len * 4;
            let fs = unsafe { transmute::<&mut Vec<_>, &mut Vec<F>>(digests_and_caps_buf) };

            unsafe {
                fs.set_len(len_with_F);
            }
            println!(
                "alllen: {}, digest_and_cap_buf_len: {}, diglen: {}",
                alllen, len_with_F, digests_and_caps_buf_len
            );
            unsafe {
                transmute::<&DeviceSlice<F>, &DeviceSlice<u64>>(
                    &values_device[alllen..alllen + len_with_F],
                )
                .async_copy_to(
                    transmute::<&mut Vec<F>, &mut Vec<u64>>(fs),
                    &ctx.inner.stream,
                )
                .unwrap();
                ctx.inner.stream.synchronize().unwrap();
            }

            unsafe {
                fs.set_len(len_with_F / 4);
            }
        });

        let coeffs = values_flatten
            .par_chunks(values_num_per_poly)
            .map(|chunk| PolynomialCoeffs {
                coeffs: chunk.to_vec(),
            })
            .collect::<Vec<_>>();

        {
            let polynomials = coeffs;
            let ctx_ext_values_flatten = ctx.ext_values_flatten.clone();
            // let ctx_ext_values_flatten :Arc<Vec<F>> = Arc::new(vec![]);
            let ctx_digests_and_caps_buf = ctx.digests_and_caps_buf3.clone();

            let mut my_leaves_dev_offset = quotient_polys_offset as isize;

            let ctx_ext_values_flatten_len = ext_values_flatten_len;
            let merkle_tree = MerkleTree {
                leaves: vec![],
                // leaves,
                // digests: digests_and_caps_buf[0..num_digests].to_vec(),
                digests: vec![],
                cap: MerkleCap(
                    ctx_digests_and_caps_buf[num_digests..num_digests_and_caps].to_vec(),
                ),
                my_leaf_len: poly_num + salt_size,
                my_leaves: ctx_ext_values_flatten,
                my_leaves_len: ctx_ext_values_flatten_len,
                my_leaves_dev_offset,
                my_digests: ctx_digests_and_caps_buf,
            };

            Self {
                polynomials,
                merkle_tree,
                degree_log: lg_n,
                rate_bits,
                blinding,
            }
        }
    }

    /// Produces a batch opening proof.
    pub fn prove_openings_with_gpu(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
        ctx: &mut Option<&mut crate::fri::oracle::CudaInvContext<F, C, D>>,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }


        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );
        println!(
            "lde_final_poly len:{}, lde_final_values len: {}",
            lde_final_poly.len(),
            lde_final_values.len()
        );

        let fri_proof = timed!(
            timing,
            "compute fri proof",
            fri_proof_with_gpu::<F, C, D>(
                &oracles
                    .par_iter()
                    .map(|c| &c.merkle_tree)
                    .collect::<Vec<_>>(),
                lde_final_poly,
                lde_final_values,
                challenger,
                fri_params,
                final_poly_coeff_len,
                max_num_query_steps,
                timing,
                ctx,
            )
        );

        fri_proof
    }
}
