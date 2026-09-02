import{E as k,m as N,s as E,r as p,as as S,j as e,g as b}from"./index-CxOmS6HQ.js";import{M as T}from"./Markdown-CPUhK-ny.js";import{D as H}from"./DiagnosticReport-CYrjTU1O.js";import G from"./getting-started-DCB6Nmue.js";import I from"./using-the-app-DAxVEW9S.js";import $ from"./DATASET_GUIDE-JA9FMUY7.js";import C from"./settings-reference-BSVEnSGe.js";import F from"./troubleshooting-r1nSbECI.js";import{m as P}from"./headingId-CD7HUh8q.js";const M=`# Getting help & reporting problems

Stuck, found a bug, or missing a feature? Two doors, both watched:

- **Discord** — [discord.gg/j6hnJBFtXE](https://discord.gg/j6hnJBFtXE) — ask in
  **#help**; usually the fastest way to get unstuck. Feature ideas and votes
  live in **#roadmap**.
- **GitHub** — [Issues](https://github.com/perfectgf/lora-dataset-studio/issues) —
  best for reproducible bugs and feature requests; the templates walk you
  through what to include.

---

## What makes a report solvable

The difference between a five-minute fix and a week of guessing is almost
always the same four things:

1. **Version** — shown in Settings → Maintenance → Updates ("Current build").
2. **Environment** — OS, and whether you run API-only, full local, or Docker.
3. **What you did → what you expected → what happened** — three short lines
   beat three paragraphs.
4. **The log** — the last lines of the server log usually name the real error.
   Settings → Maintenance → 🪵 Server log → **Copy all**.

## Or let the app write it for you

The **diagnostic report** button below assembles all of that in one click:
version, OS, capability status, non-secret settings and the last log lines —
formatted, copied to your clipboard, ready to paste into Discord or a GitHub
issue.

What it deliberately **never** includes: your API keys or tokens (only
whether each one is set) and your folder paths (only whether each one is
configured). One caveat: the log tail can mention file names from your machine
— skim the paste before posting if that matters to you.

## Feature requests

Describe the **job you were doing when you missed the feature** — the problem
is more valuable than the proposed solution. Post it in Discord **#roadmap** or
open a GitHub issue with the *Feature request* template.

## Support the project

LoRA Dataset Studio is free, open source and built in the open. If it saves
you time and you want to help development, you can sponsor it on
[GitHub Sponsors](https://github.com/sponsors/perfectgf) — one-time or
monthly, and 100% of it goes to the project (GitHub charges no fees).
The best free ways to help are just as welcome: report bugs, share ideas on
Discord, and star the repo.
`,h=[{id:"getting-started",num:"01",title:"Getting started",description:"Install the app, connect the tools you need, and understand the workspace.",source:G},{id:"using-the-app",num:"02",title:"Using the app",description:"Follow the complete workflow for character, concept, and style datasets.",source:I},{id:"dataset-guide",num:"03",title:"Building a good dataset",description:"Make stronger choices about images, captions, settings, and checkpoints.",source:$},{id:"settings-reference",num:"04",title:"Settings reference",description:"Every setting explained — what it does, its default, and when to change it.",source:C},{id:"troubleshooting",num:"05",title:"Troubleshooting",description:"Find a symptom, understand the cause, and apply the shortest reliable fix.",source:F}],A={id:"getting-help",num:"06",title:"Getting help",description:"Create a useful report and share the details needed to solve a problem.",source:M,extra:"diagnostic"},D=n=>n.replace(/[`*_]/g,""),R=n=>n.focus?`${n.route}${n.route.includes("?")?"&":"?"}focus=${n.focus}`:n.route;function X({helpOnly:n=!1}){const{section:f}=k(),l=N(),[v]=E(),d=v.get("h"),a=n?[A]:h,i=n?0:Math.max(0,a.findIndex(t=>t.id===f)),s=a[i],c=i>0?a[i-1]:null,u=i<a.length-1?a[i+1]:null,m=[...s.source.matchAll(/^##\s+(.+)$/gm)].map(t=>({title:D(t[1]),id:P(t[1])})),j=Math.max(1,Math.ceil(s.source.trim().split(/\s+/).length/210)),g=t=>{var r;return(r=document.getElementById(t))==null?void 0:r.scrollIntoView({behavior:"smooth",block:"start"})},w=p.useMemo(()=>{const t={};for(const r of S(s.id))t[r.guide.anchor]||(t[r.guide.anchor]=e.jsx("button",{type:"button",onClick:()=>l(R(r.app)),className:"inline-flex items-center gap-1 whitespace-nowrap rounded-md border border-indigo-400/40 bg-indigo-500/10 px-2.5 py-1 text-xs font-medium text-indigo-200 transition-colors hover:bg-indigo-500/20",children:"Open this screen →"}));return t},[s.id,l]);p.useEffect(()=>{d||window.scrollTo(0,0)},[s.id,d]),p.useEffect(()=>{if(!d)return;const t=document.getElementById(d);if(!t)return;t.scrollIntoView({behavior:"smooth",block:"start"});const r=["ring-2","ring-indigo-400/70","ring-offset-2","ring-offset-app"];t.classList.add(...r);const o=setTimeout(()=>t.classList.remove(...r),2e3);return()=>clearTimeout(o)},[d,s.id]);const x=(t,r)=>{const o=t.id===s.id,y=r?`flex shrink-0 items-baseline gap-1.5 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs font-medium ${o?"border-border-strong bg-surface-raised text-content":"border-border text-content-muted hover:text-content"}`:`relative flex w-full items-baseline gap-2.5 rounded-md px-3 py-2 text-left text-sm ${o?"bg-surface-raised text-content":"text-content-muted hover:bg-surface hover:text-content"}`;return e.jsxs("button",{type:"button",onClick:()=>l(`/guide/${t.id}`),"aria-current":o?"page":void 0,className:y,children:[!r&&o&&e.jsx("span",{"aria-hidden":!0,className:"absolute bottom-1.5 left-0 top-1.5 w-0.5 rounded bg-gradient-primary"}),e.jsx("span",{className:`font-mono text-[11px] ${o?"text-content":"text-content-subtle"}`,children:t.num}),e.jsx("span",{className:"font-medium",children:t.title})]},t.id)};return e.jsxs("div",{className:n?"mx-auto max-w-5xl xl:grid xl:grid-cols-[minmax(0,1fr)_190px] xl:items-start xl:gap-7":"lg:grid lg:grid-cols-[210px_minmax(0,1fr)] lg:items-start lg:gap-7 xl:grid-cols-[210px_minmax(0,1fr)_190px]",children:[!n&&e.jsxs("aside",{children:[e.jsx("nav",{"aria-label":"Guide chapters",className:"relative -mx-4 flex gap-2 overflow-x-auto px-4 pb-3 lg:hidden",children:h.map(t=>x(t,!0))}),e.jsxs("nav",{"aria-label":"Guide chapters",className:"hidden lg:sticky lg:top-20 lg:block",children:[e.jsx("p",{className:"px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle",children:"Field manual"}),e.jsx("div",{className:"flex flex-col gap-0.5",children:h.map(t=>x(t,!1))})]})]}),e.jsxs("main",{className:`min-w-0 max-w-4xl pb-10 ${n?"mx-auto":"mt-2 lg:mt-0"}`,children:[e.jsxs("header",{className:"relative mb-4 overflow-hidden rounded-2xl border border-border bg-surface px-5 py-5 sm:px-6 sm:py-6",children:[e.jsx("div",{"aria-hidden":!0,className:"absolute -right-16 -top-20 h-52 w-52 rounded-full bg-indigo-500/10 blur-3xl"}),e.jsxs("div",{className:"relative",children:[e.jsxs("div",{className:"mb-3 flex flex-wrap items-center gap-2 font-mono text-[0.6875rem] uppercase tracking-[0.14em] text-content-subtle",children:[e.jsx("span",{className:"rounded-md border border-indigo-400/30 bg-indigo-500/10 px-2 py-1 text-indigo-300",children:n?"Support":`Chapter ${s.num}`}),e.jsxs("span",{children:[j," min read"]}),!n&&e.jsxs(e.Fragment,{children:[e.jsx("span",{"aria-hidden":!0,children:"·"}),e.jsxs("span",{children:[i+1," of ",a.length]})]})]}),e.jsx("h1",{className:"m-0 max-w-2xl text-2xl font-bold tracking-tight text-content sm:text-3xl",children:s.title}),e.jsx("p",{className:"mb-0 mt-2 max-w-2xl text-sm leading-relaxed text-content-muted sm:text-base",children:s.description})]})]}),m.length>0&&e.jsxs("nav",{"aria-label":"On this page",className:"mb-4 rounded-xl border border-border bg-surface p-3 xl:hidden",children:[e.jsx("p",{className:"m-0 mb-2 font-mono text-[0.625rem] uppercase tracking-[0.16em] text-content-subtle",children:"On this page"}),e.jsx("div",{className:"flex gap-2 overflow-x-auto pb-0.5",children:m.map(t=>e.jsx("button",{type:"button",onClick:()=>g(t.id),className:"shrink-0 rounded-full border border-border bg-transparent px-2.5 py-1 text-xs text-content-muted hover:border-border-strong hover:text-content",children:t.title},t.id))})]}),e.jsx(T,{source:s.source,variant:"guide",sectionActions:w}),s.extra==="diagnostic"&&e.jsx("div",{className:"mt-6",children:e.jsx(H,{})}),!n&&e.jsxs("div",{className:"mt-6 grid grid-cols-2 gap-3 border-t border-border pt-4",children:[c?e.jsxs(b,{to:`/guide/${c.id}`,className:"group flex min-w-0 items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 no-underline hover:bg-surface-raised",children:[e.jsx("span",{"aria-hidden":!0,className:"text-content-subtle",children:"←"}),e.jsxs("span",{className:"min-w-0",children:[e.jsx("span",{className:"block font-mono text-[0.625rem] uppercase tracking-wider text-content-subtle",children:"Previous"}),e.jsx("span",{className:"block truncate text-sm font-medium text-content-muted group-hover:text-content",children:c.title})]})]}):e.jsx("span",{}),u?e.jsxs(b,{to:`/guide/${u.id}`,className:"group flex min-w-0 items-center justify-end gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 text-right no-underline hover:bg-surface-raised",children:[e.jsxs("span",{className:"min-w-0",children:[e.jsx("span",{className:"block font-mono text-[0.625rem] uppercase tracking-wider text-content-subtle",children:"Next"}),e.jsx("span",{className:"block truncate text-sm font-medium text-content-muted group-hover:text-content",children:u.title})]}),e.jsx("span",{"aria-hidden":!0,className:"text-content-subtle",children:"→"})]}):e.jsx("span",{})]})]}),e.jsx("aside",{className:"hidden xl:block",children:e.jsxs("nav",{"aria-label":"On this page",className:"sticky top-20 border-l border-border pl-4",children:[e.jsx("p",{className:"m-0 mb-2 font-mono text-[0.625rem] uppercase tracking-[0.16em] text-content-subtle",children:"On this page"}),e.jsx("div",{className:"flex flex-col gap-0.5",children:m.map(t=>e.jsx("button",{type:"button",onClick:()=>g(t.id),className:"rounded-md bg-transparent px-2 py-1.5 text-left text-xs leading-snug text-content-subtle hover:bg-surface hover:text-content",children:t.title},t.id))})]})})]})}export{X as default};
