#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
namespace cpu=strata::kernels::cpu;
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"gguf layer experts workers host_works tasks\n";return 2;}
 strata::GgufFile file(argv[1]);int layer=std::stoi(argv[2]),n=std::stoi(argv[3]),workers=std::stoi(argv[4]);bool host=std::stoi(argv[5]);int tasks=std::stoi(argv[6]);
 if(n<1||n>16)return 2;
 const strata::TensorInfo *g=nullptr,*u=nullptr,*d=nullptr;
 for(auto& t:file.tensors()){std::string p="blk."+std::to_string(layer)+".ffn_";if(t.name==p+"gate_exps.weight")g=&t;if(t.name==p+"up_exps.weight")u=&t;if(t.name==p+"down_exps.weight")d=&t;}
 if(!g||!u||!d)return 3;
 cpu::NativeFmt f;std::string err;if(!cpu::native_fmt(g->type,d->type,2560,640,f,err)){std::cerr<<err;return 4;}
 std::vector<uint8_t> blobs(n*f.bytes);std::vector<float>x(2560),out(n*2560),ref;for(int i=0;i<2560;i++)x[i]=std::sin(float(i)*.01f);
 alignas(64) uint8_t act[cpu::kNativeActBytes];cpu::native_quant_act(f,x.data(),act);
 std::vector<cpu::ExpertJobMulti> jobs(n);
 for(int j=0;j<n;j++){size_t e=(j*37+5)%256;auto*b=blobs.data()+j*f.bytes;memcpy(b,file.tensor_data(*g)+e*f.up_off,f.up_off);memcpy(b+f.up_off,file.tensor_data(*u)+e*f.up_off,f.up_off);memcpy(b+f.down_off,file.tensor_data(*d)+e*(f.bytes-f.down_off),f.bytes-f.down_off);jobs[j].blob=b;jobs[j].nt=1;jobs[j].nact[0]=act;jobs[j].out[0]=out.data()+j*2560;}
 cpu::ExpertPool pool(workers,false,host,cpu::PoolAffinity::All,tasks);
 std::vector<double> times;
 for(int k=0;k<120;k++){auto t=std::chrono::steady_clock::now();pool.run_split_multi_native(f,jobs.data(),n);double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();if(k==0)ref=out;else if(memcmp(ref.data(),out.data(),out.size()*4)){std::cerr<<"output mismatch";return 5;}if(k>=20)times.push_back(us);}
 double sum=0;for(float v:out){if(!std::isfinite(v))return 6;sum+=v;}
 std::sort(times.begin(),times.end());std::cout<<layer<<","<<n<<","<<workers<<","<<host<<","<<tasks<<","<<times[50]<<","<<times[90]<<","<<times[99]<<","<<sum<<"\n";
}
