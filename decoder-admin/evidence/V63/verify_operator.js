const fs=require('fs'),vm=require('vm');
(async()=>{
const source=fs.readFileSync(__dirname+'/OPERATOR.js.txt','utf8'),cases=[];
for(const [name,outcomes] of [['drain_lock_success',[76,75,0]],['unexpected_stop',[1]],['bounded_transients',Array(9).fill(76)]]){
 const counts={launch:0,monitor:0,target:0,registration:0,fault:0,status:0};let reported=null;const state=new Map([['v63catalog',[]],['v63readbacks',[]]]);
 const context={store:(k,v)=>state.set(k,v),load:k=>state.get(k),notify:()=>{},text:x=>{reported=x;},setTimeout:f=>{f();},tools:{
 exec_command:async({cmd})=>{
  if(cmd.includes('/admin.py ')){counts.status++;return {exit_code:0,output:JSON.stringify({objects:[],metrics:{}})};}
  if(cmd.includes('/run_native.py')){counts.launch++;return {session_id:101};}
  if(cmd.includes('/observe_lifecycle.py')){counts.monitor++;return {session_id:102};}
  if(cmd.includes('/prepare_target.py')){counts.target++;return {exit_code:0,output:'{}'};}
  if(cmd.includes('/inject_once.py')){const code=outcomes[counts.fault++];return {exit_code:code,output:''};}
  if(cmd.includes('test -f')&&cmd.includes('RUN_FINISHED'))return {exit_code:1};
  return {exit_code:0,output:''};
 },mcp__codex_apps__google_drive_upload_file:async()=>{counts.registration++;return {structuredContent:{success:true,id:'SYNTHETIC'}};}}};
 let error=null;try{await vm.runInNewContext('('+source+')()',context);}catch(e){error=e.message;}
 if(counts.launch!==1||counts.monitor!==1||counts.target!==1||counts.registration!==1)throw Error('Repeated launch/target/registration');
 if(name==='drain_lock_success'&&(counts.fault!==3||reported.fault.exit_code!==0))throw Error('Retry sequence');
 if(name==='unexpected_stop'&&(counts.fault!==1||reported.fault.exit_code!==1))throw Error('Unexpected failure retried');
 if(name==='bounded_transients'&&(counts.fault!==8||error!=='PREMUTATION_ADMISSION_BOUND'))throw Error('Unbounded retry');
 cases.push({name,counts,error,pass:true});
}
fs.writeFileSync(__dirname+'/OPERATOR_VERIFICATION.json',JSON.stringify({scope:'mock connector/control flow; no decoder execution or real mutation',cases},null,2)+'\n');console.log(JSON.stringify(cases));
})().catch(e=>{console.error(e);process.exit(1)});
