import{c as t}from"./index-Ci9Jqa3s.js";/**
 * @license lucide-react v1.34.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const d=[["rect",{width:"12",height:"12",x:"2",y:"10",rx:"2",ry:"2",key:"6agr2n"}],["path",{d:"m17.92 14 3.5-3.5a2.24 2.24 0 0 0 0-3l-5-4.92a2.24 2.24 0 0 0-3 0L10 6",key:"1o487t"}],["path",{d:"M6 18h.01",key:"uhywen"}],["path",{d:"M10 14h.01",key:"ssrbsk"}],["path",{d:"M15 6h.01",key:"cblpky"}],["path",{d:"M18 9h.01",key:"2061c0"}]],r=t("dices",d),s=o=>Number(o).toFixed(2).replace(/0$/,"").replace(/\.$/,".0");function n(o){const l=o||{},i=(l.local_llm||{}).provider||"ollama";if(i==="lmstudio"){const e=l.lmstudio||{};return{provider:i,installed:!!e.reachable,reachable:!!e.reachable,vision_model_ready:!!e.model_ready,vision_model:e.vision_model||"",detail:e.detail||""}}const a=l.ollama||{};return{provider:i,installed:!!a.installed,reachable:!!a.reachable,vision_model_ready:!!a.vision_model_ready,vision_model:a.vision_model||"",detail:""}}export{r as D,n as a,s as f};
