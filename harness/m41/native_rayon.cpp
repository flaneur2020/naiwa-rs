#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
extern "C" void bench_rayon_init(size_t);
extern "C" void bench_rayon_run(size_t,size_t,void(*)(size_t,size_t));
namespace cpu=strata::kernels::cpu;
struct Phase { const cpu::NativeFmt* f; std::vector<cpu::ExpertJobMulti>* jobs; float* ff; uint8_t* hq; int n,tasks,nt; bool down; };
extern "C" void native_task(size_t ctx,size_t task) {
 auto& p=*reinterpret_cast<Phase*>(ctx);int per=p.down?2560:640;int64_t end=int64_t(p.n)*per*(task+1)/p.tasks;
 for(int64_t r=int64_t(p.n)*per*task/p.tasks;r<end;){int e=r/per,r0=r%per,r1=std::min<int64_t>(per,r0+end-r);auto&j=(*p.jobs)[e];
 if(p.down){const void* h[cpu::MAXT];for(int t=0;t<p.nt;t++)h[t]=p.hq+(e*p.nt+t)*cpu::kNativeHBytes;cpu::native_down_rows(*p.f,j.blob,h,p.nt,j.out,r0,r1);}
 else{float* f[cpu::MAXT];for(int t=0;t<p.nt;t++)f[t]=p.ff+(e*p.nt+t)*640;cpu::native_gu_rows(*p.f,j.blob,j.nact,p.nt,f,r0,r1);} r+=r1-r0;}
}
int main(int argc,char**argv){
 if(argc!=9){std::cerr<<"gguf layer experts workers host_works tasks rows gap_us\n";return 2;}
 strata::GgufFile file(argv[1]);int layer=std::stoi(argv[2]),n=std::stoi(argv[3]),workers=std::stoi(argv[4]);bool host=std::stoi(argv[5]);int tasks=std::stoi(argv[6]);
 int nt=std::stoi(argv[7]),gap=std::stoi(argv[8]);if(n<1||n>16||nt<1||nt>cpu::MAXT||gap<0||tasks<1||tasks>n*640)return 2;
 const strata::TensorInfo *g=nullptr,*u=nullptr,*d=nullptr;
 for(auto& t:file.tensors()){std::string p="blk."+std::to_string(layer)+".ffn_";if(t.name==p+"gate_exps.weight")g=&t;if(t.name==p+"up_exps.weight")u=&t;if(t.name==p+"down_exps.weight")d=&t;}
 if(!g||!u||!d)return 3;
 cpu::NativeFmt f;std::string err;if(!cpu::native_fmt(g->type,d->type,2560,640,f,err)){std::cerr<<err;return 4;}
 std::vector<uint8_t> blobs(n*f.bytes);std::vector<float>x(nt*2560),out(n*nt*2560),ref;for(int i=0;i<nt*2560;i++)x[i]=std::sin(float(i)*.01f);
 alignas(64) uint8_t act[cpu::MAXT][cpu::kNativeActBytes];for(int t=0;t<nt;t++)cpu::native_quant_act(f,x.data()+t*2560,act[t]);
 std::vector<cpu::ExpertJobMulti> jobs(n);
 for(int j=0;j<n;j++){size_t e=(j*37+5)%256;auto*b=blobs.data()+j*f.bytes;memcpy(b,file.tensor_data(*g)+e*f.up_off,f.up_off);memcpy(b+f.up_off,file.tensor_data(*u)+e*f.up_off,f.up_off);memcpy(b+f.down_off,file.tensor_data(*d)+e*(f.bytes-f.down_off),f.bytes-f.down_off);jobs[j].blob=b;jobs[j].nt=nt;for(int t=0;t<nt;t++){jobs[j].nact[t]=act[t];jobs[j].out[t]=out.data()+(j*nt+t)*2560;}}
 bool rayon=std::getenv("BENCH_RAYON")!=nullptr;
 if(f.gu_type==42||f.d_type==42){std::cerr<<"Q2 specialized path not implemented";return 7;}
 std::vector<float> ff(n*nt*640);std::vector<uint8_t> hq(n*nt*cpu::kNativeHBytes);
 Phase phase{&f,&jobs,ff.data(),hq.data(),n,tasks,nt,false};
 std::unique_ptr<cpu::ExpertPool> pool;
 {cpu::ExpertPool reference(5,false,true,cpu::PoolAffinity::All,tasks);reference.run_split_multi_native(f,jobs.data(),n);ref=out;}
 if(rayon)bench_rayon_init(workers);else pool=std::make_unique<cpu::ExpertPool>(workers,false,host,cpu::PoolAffinity::All,tasks);
 auto run=[&]{if(!rayon){pool->run_split_multi_native(f,jobs.data(),n);return;}phase.down=false;bench_rayon_run(tasks,reinterpret_cast<size_t>(&phase),native_task);for(int e=0;e<n*nt;e++)cpu::native_quant_h(f,ff.data()+e*640,hq.data()+e*cpu::kNativeHBytes);phase.down=true;bench_rayon_run(tasks,reinterpret_cast<size_t>(&phase),native_task);};
 std::vector<double> times;
 for(int k=0;k<120;k++){if(gap)std::this_thread::sleep_for(std::chrono::microseconds(gap));auto t=std::chrono::steady_clock::now();run();double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();if(memcmp(ref.data(),out.data(),out.size()*4)){std::cerr<<"output mismatch";return 5;}if(k>=20)times.push_back(us);}
 double sum=0;for(float v:out){if(!std::isfinite(v))return 6;sum+=v;}
 std::sort(times.begin(),times.end());std::cout<<layer<<","<<n<<","<<workers<<","<<host<<","<<tasks<<","<<nt<<","<<gap<<","<<times[50]<<","<<times[90]<<","<<times[99]<<","<<sum<<"\n";
}
