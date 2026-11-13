use rayon::prelude::*;
use std::sync::OnceLock;
static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
#[no_mangle]
pub extern "C" fn bench_rayon_init(n: usize) { POOL.set(rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap()).ok().unwrap(); }
// Caller owns the context until install returns; tasks must write disjoint output ranges.
#[no_mangle]
pub unsafe extern "C" fn bench_rayon_run(n: usize, context: usize, task: unsafe extern "C" fn(usize,usize)) {
    POOL.get().unwrap().install(|| (0..n).into_par_iter().for_each(|i| unsafe {task(context,i)}));
}
