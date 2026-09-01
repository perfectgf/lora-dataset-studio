import{c as n}from"./index-llxYDHRq.js";/**
 * @license lucide-react v1.34.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const c=[["path",{d:"m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2",key:"usdka0"}]],l=n("folder-open",c);function i({response:r,thrown:e,declined:s,fallback:t="Request failed",clean:a}={}){const u=o=>a?a(o,t):String(o??"").trim()||t;if(s)return{close:!1,error:null};if(e)return{close:!1,error:u((e==null?void 0:e.message)??e)};if(!r)return{close:!1,error:`${t} — no answer from the server.`};if(r.ok===!1){const o=u(r.error);return{close:!1,error:r.hint?`${o} — ${r.hint}`:o}}return{close:!0,error:null}}async function m(r,{fallback:e,clean:s}={}){try{return i({response:await r(),fallback:e,clean:s})}catch(t){return i({thrown:t,fallback:e,clean:s})}}export{l as F,m as a,i as s};
