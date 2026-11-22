"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="cs2-token"]').content;
const names = {running:"训练中",completed:"已完成",failed:"失败",cancelled:"已停止",unknown:"状态待确认",unconfirmed:"未知 · 已归档",idle:"空闲"};
let snapshot = null, selectedJob = null, busy = false, logText = null, initialPresets = false;
const format = x => typeof x === "number" && Number.isFinite(x) ? x.toFixed(4) : "—";
const number = x => typeof x === "number" ? x.toLocaleString("zh-CN") : "—";
const date = x => x ? new Date(x).toLocaleString("zh-CN", {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false}) : "—";
function node(tag, text, cls) { const n=document.createElement(tag); if(text !== undefined)n.textContent=text; if(cls)n.className=cls; return n; }
function error(message) { $("error").hidden=!message; $("error").textContent=message || ""; }
async function post(path, data) {
  const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-CS2-Token":token},body:JSON.stringify(data)});
  const result=await response.json(); if(!response.ok)throw new Error(result.error || "请求失败"); return result;
}
function renderMetrics(job) {
  let values=snapshot.corpus?.test?.calibrated || {};
  let title="Cologne · 已完成参考结果", subtitle="校准模型 · 独立赛事评估（历史）";
  if(job){
    const events=(job.events || []).filter(e=>e.details?.metrics);
    const validation=events.findLast(e=>e.details.split==="validation");
    values=job.report?.metrics?.validation?.model || validation?.details.metrics || {};
    const preset=snapshot.presets.find(p=>p.id===job.preset);
    title="当前任务 · 验证集指标"; subtitle=Object.keys(values).length ? (preset?.validation_name || "验证集")+" · 回合加权" : "验证阶段完成后更新";
  }
  $("metric-title").textContent=title; $("metric-subtitle").textContent=subtitle;
  $("metric-loss").textContent=format(values.log_loss); $("metric-brier").textContent=format(values.brier_score);
  $("metric-auc").textContent=format(values.roc_auc);
  let ece=values.ece_10_bins;
  if(ece === undefined && values.calibration_bins?.length)ece=values.calibration_bins.reduce((s,b)=>s+b.weight*Math.abs(b.mean_probability-b.ct_win_frequency),0)/values.round_weight_sum;
  $("metric-ece").textContent=format(ece);
}
function renderJob() {
  const jobs=snapshot.jobs || [];
  const active=jobs.find(j=>j.status==="running");
  const job=jobs.find(j=>j.id===selectedJob) || active || jobs[0];
  const state=job?.status || "idle";
  $("run-status").className="badge "+state; $("run-status").textContent=names[state] || state;
  $("run-name").textContent=job?.name || "准备开始一次训练";
  $("run-description").textContent=job ? job.id+" · "+job.dataset : "从已验证的数据集启动，原有模型会保留。";
  let seconds=job ? Math.max(0,Math.floor(((job.finished_at ? new Date(job.finished_at) : state==="running" ? new Date() : new Date(job.events.at(-1)?.time || job.started_at))-new Date(job.started_at))/1000)) : 0;
  $("elapsed").textContent=String(Math.floor(seconds/60)).padStart(2,"0")+":"+String(seconds%60).padStart(2,"0");
  const events=job?.events || [], stages=snapshot.stages || [];
  const completed=stages.filter(s=>events.some(e=>e.stage===s.id&&e.state==="completed")).length;
  $("stage-count").textContent=completed+" / "+stages.length;
  $("progress").style.width=(stages.length ? completed/stages.length*100 : 0)+"%";
  $("iterations").textContent=events.findLast(e=>e.details?.iterations)?.details.iterations ?? "—";
  $("stage-message").textContent=state==="unknown" ? "服务重启后无法重连任务；可查看日志或归档记录，归档不会终止原进程。" : job?.error || (state==="cancelled" ? "任务已停止；未完成的输出不会标为成功。" : events.at(-1)?.message) || (state==="running" ? "训练进程已启动，等待数据校验。" : "当前没有正在运行的任务。");
  $("stages").replaceChildren(...stages.map((stage,index)=>{
    const last=events.findLast(e=>e.stage===stage.id);
    const stageState=last?.state==="completed" ? "completed" : last?.state==="active"&&state==="running" ? "active" : "";
    const el=node("div",undefined,"stage "+stageState); el.append(node("span",stageState==="completed"?"✓":String(index+1),"stage-circle"),node("span",stage.name)); return el;
  }));
  const log=job?.log || (state==="running" ? "进程已启动，等待首条日志。" : "等待启动训练。这里会显示实际进程输出。");
  if(log!==logText){$("logs").textContent=log;logText=log;if($("autoscroll").checked)$("logs").scrollTop=$("logs").scrollHeight;}
  $("output-path").textContent=job?.output || "模型会保存到独立的新目录";
  $("log-state").textContent=state.toUpperCase();
  const canStart=!jobs.some(j=>["running","unknown"].includes(j.status));
  const preset=snapshot.presets.find(p=>p.id===$("preset").value);
  $("start").disabled=busy || !canStart || !preset?.available;
  $("start").textContent=active ? "训练进行中…" : "▶  开始训练";
  $("preset").disabled=busy || !!active;
  $("features").textContent=preset?.features || "—";
  $("algorithm").textContent=preset?.algorithm || "Logistic Regression";
  $("regularization-label").textContent=preset?.regularization_label || "逻辑回归 C";
  $("regularization-value").textContent=preset?.regularization_value || "1.0";
  $("config-note").textContent=(preset?.description || "数据集配置不可用。")+" Bucharest 保留集未加入。";
  $("cancel").hidden=!active; $("cancel").disabled=busy;
  $("archive").hidden=!jobs.some(j=>j.status==="unknown"); $("archive").disabled=busy;
  renderMetrics(job);
}
function renderHistory() {
  const tbody=$("history-rows"), jobs=snapshot.jobs || [];
  const rows=jobs.map(job=>{
    const tr=node("tr");tr.dataset.id=job.id;tr.title="查看此任务的阶段与日志";
    const preset=snapshot.presets.find(p=>p.id===job.preset);
    const label=node("td",job.name);label.append(node("small",job.id+(preset?.test_name ? " · 测试："+preset.test_name : "")));
    const status=node("td");status.append(node("span",names[job.status] || job.status,"badge "+job.status));
    const score=job.report?.metrics?.test?.model || (job.preset==="phase2_temporal_validation" ? job.report?.metrics?.validation?.model : null) || {};
    tr.append(label,status,node("td",format(score.log_loss)),node("td",format(score.brier_score)),node("td",date(job.started_at)));
    tr.onclick=()=>{selectedJob=job.id;renderJob();$("monitor").scrollIntoView({behavior:"smooth"});};return tr;
  });
  const jobPaths=new Set(jobs.map(j=>j.output+"/report.json"));
  for(const report of snapshot.history || []){
    if(jobPaths.has(report.path))continue;
    const tr=node("tr"), label=node("td",report.name);
    const detail=report.mode==="smoke_test"?"流程检查 · 无独立评分":report.mode==="additional_event_holdout"?"额外赛事留出评估 · "+report.event:report.mode==="temporal_historical_validation"?"波托选定模型 · 新增历史验证":"已观察的历史评估";
    label.append(node("small",detail));
    const status=node("td");status.append(node("span","历史结果","badge idle"));
    tr.append(label,status,node("td",format(report.scores?.log_loss)),node("td",format(report.scores?.brier_score)),node("td",date(report.created_at)));rows.push(tr);
  }
  if(!rows.length){const tr=node("tr"),td=node("td","暂无实验记录");td.colSpan=5;tr.append(td);rows.push(tr);}
  tbody.replaceChildren(...rows);
}
function renderBatch(batch={}, prefix="batch"){
  const item=suffix=>$(prefix+"-"+suffix), panel=item("panel");
  panel.hidden=!batch.total;if(panel.hidden)return;
  const ok=batch.completed?.length || 0, failed=batch.failed?.length || 0, total=batch.total;
  item("count").textContent=(ok+failed)+" / "+total+" 场已处理";
  item("progress").style.width=(Math.min(1,(ok+failed)/total)*100)+"%";
  const state=batch.status==="completed"?"已完成":batch.status==="completed_with_errors"?"部分失败":"处理中";
  item("status").className="badge "+(batch.status==="completed"?"completed":batch.status==="completed_with_errors"?"failed":"running");
  item("status").textContent=state;
  item("current").textContent=batch.current?"当前："+batch.current.matchup+" · "+batch.current.id:(ok+failed<total?"等待下一场":"数据处理已结束");
  item("detail").textContent="验证通过 "+ok+" 场 · 待处理 "+Math.max(0,total-ok-failed)+" 场 · 失败 "+failed+" 场 · 最近更新 "+date(batch.updated_at_utc);
}
function renderCorpus(){
  const preset=snapshot.presets.find(p=>p.id===$("preset").value) || {}, counts=preset.match_counts || {};
  const matches=Object.values(counts).reduce((s,v)=>s+v,0), samples=preset.sample_count || 0;
  const facts=[["比赛",matches],["有效回合",preset.round_count],["采样时点",samples],["玩家状态",samples*10]];
  $("corpus-counts").replaceChildren(...facts.map(([label,value])=>{const el=node("div");el.append(node("span",label),node("b",value?number(value):"—"));return el;}));
  $("split-summary").replaceChildren(...Object.entries(counts).map(([split,count])=>{const el=node("span");el.append(node("i"),document.createTextNode(split+"  "+count+" 场"));return el;}));
}
function renderContext(){
  const batch=snapshot.context_batch || {}, panel=$("context-panel");
  panel.hidden=!batch.total;if(panel.hidden)return;
  const ok=batch.completed || 0, failed=batch.failed || 0, total=batch.total;
  $("context-count").textContent=(ok+failed)+" / "+total+" 个录制文件已检查";
  $("context-progress").style.width=(Math.min(1,(ok+failed)/total)*100)+"%";
  const state=batch.status==="completed"?"已完成":batch.status==="completed_with_errors"?"部分失败":"处理中";
  $("context-status").className="badge "+(batch.status==="completed"?"completed":batch.status==="completed_with_errors"?"failed":"running");
  $("context-status").textContent=state;
  $("context-current").textContent=batch.current ? "当前："+batch.current : "上下文核对已结束";
  $("context-detail").textContent="完成 "+ok+" 个 · 仅事件回退 "+(batch.event_only ?? "—")+" 个 · 待处理 "+Math.max(0,total-ok-failed)+" 个 · 失败 "+failed+" 个 · 最近更新 "+date(batch.updated_at_utc);
}
async function refresh(){
  try{
    const response=await fetch("/api/state",{cache:"no-store"});if(!response.ok)throw new Error("读取状态失败");
    snapshot=await response.json();
    if(!initialPresets){$("preset").replaceChildren(...snapshot.presets.map(p=>{const option=node("option",p.name+(p.available?"":"（未构建）"));option.value=p.id;option.disabled=!p.available;return option;}));initialPresets=true;}
    $("connection").textContent="本地已连接";$("connection-dot").style.background="var(--accent)";
    $("updated").textContent="更新于 "+new Date().toLocaleTimeString("zh-CN",{hour12:false});
    renderJob();renderHistory();renderBatch(snapshot.holdout_batch);renderBatch(snapshot.historical_validation_batch,"historical");renderContext();renderCorpus();
  }catch(e){$("connection").textContent="连接中断";$("connection-dot").style.background="#e7a17f";$("start").disabled=true;$("cancel").disabled=true;}
}
$("preset").onchange=()=>{if(snapshot){renderJob();renderCorpus();}};
$("start").onclick=async()=>{if(busy)return;busy=true;error("");renderJob();try{const job=await post("/api/start",{preset:$("preset").value});selectedJob=job.id;await refresh();}catch(e){error(e.message);}finally{busy=false;renderJob();}};
$("cancel").onclick=async()=>{const active=snapshot?.jobs.find(j=>j.status==="running");if(!active||busy)return;busy=true;renderJob();try{await post("/api/cancel",{id:active.id});await refresh();}catch(e){error(e.message);}finally{busy=false;renderJob();}};
$("archive").onclick=async()=>{const unknown=snapshot?.jobs.find(j=>j.status==="unknown");if(!unknown||busy)return;busy=true;renderJob();try{await post("/api/archive",{id:unknown.id});await refresh();}catch(e){error(e.message);}finally{busy=false;renderJob();}};
document.querySelectorAll(".nav").forEach(link=>link.addEventListener("click",()=>{document.querySelectorAll(".nav").forEach(n=>n.classList.remove("active"));link.classList.add("active");}));
async function poll(){await refresh();setTimeout(poll,1000);}poll();
