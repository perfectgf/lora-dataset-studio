/* Renderer markdown minimal, SANS dépendance — couvre exactement ce que
   docs/DATASET_GUIDE.md utilise (h1-h3, paragraphes, listes ± cases à cocher,
   tables, blockquotes, code fences, hr, gras/italique/code/liens inline).
   On garde le bundle distribué léger plutôt que d'embarquer react-markdown
   pour une seule page. Si le guide adopte un jour une syntaxe non couverte,
   elle sortira en texte brut — visible d'un coup d'œil, jamais dangereux
   (aucun dangerouslySetInnerHTML : tout passe par des éléments React). */

// ---- inline: **bold**, *italic*, `code`, [text](url) ----
function renderInline(text, keyBase = 'i') {
  const out = [];
  // tokenise par priorité : code d'abord (son contenu est littéral), puis gras,
  // italique, lien. Regex global unique → un seul passage gauche→droite.
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(\[[^\]]+\]\([^)]+\))/g;
  let last = 0, m, k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    const key = `${keyBase}-${k++}`;
    if (tok.startsWith('`')) {
      out.push(<code key={key} className="px-1 py-0.5 rounded bg-surface-raised text-indigo-200 text-[0.8125em] font-mono">{tok.slice(1, -1)}</code>);
    } else if (tok.startsWith('**')) {
      out.push(<strong key={key} className="text-content font-semibold">{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith('*')) {
      out.push(<em key={key}>{tok.slice(1, -1)}</em>);
    } else {
      const mm = tok.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      out.push(<a key={key} href={mm[2]} target="_blank" rel="noreferrer" className="text-indigo-300 underline decoration-indigo-400/40 hover:decoration-indigo-300">{mm[1]}</a>);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// ---- blocs ----
function parseBlocks(md) {
  const lines = md.replace(/\r\n/g, '\n').split('\n');
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith('```')) {                       // code fence
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].startsWith('```')) buf.push(lines[i++]);
      i++;                                              // skip closing fence
      blocks.push({ t: 'code', body: buf.join('\n') });
      continue;
    }
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) { blocks.push({ t: `h${h[1].length}`, body: h[2] }); i++; continue; }
    if (/^(-{3,}|\*{3,})\s*$/.test(line)) { blocks.push({ t: 'hr' }); i++; continue; }
    if (line.startsWith('>')) {                         // blockquote (multi-ligne)
      const buf = [];
      while (i < lines.length && lines[i].startsWith('>')) buf.push(lines[i++].replace(/^>\s?/, ''));
      blocks.push({ t: 'quote', body: buf.join(' ') });
      continue;
    }
    if (/^\|/.test(line)) {                             // table
      const rows = [];
      while (i < lines.length && /^\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const header = cells(rows[0]);
      // ligne 2 = séparateur |---|--- ; le corps commence après
      const body = rows.slice(2).map(cells);
      blocks.push({ t: 'table', header, body });
      continue;
    }
    if (/^(\s*)([-*]|\d+\.)\s+/.test(line)) {           // liste (± cases [ ]/[x], ± ordonnée)
      const items = [];
      const ordered = /^\s*\d+\./.test(line);
      while (i < lines.length && /^(\s*)([-*]|\d+\.)\s+/.test(lines[i])) {
        let item = lines[i].replace(/^(\s*)([-*]|\d+\.)\s+/, '');
        i++;
        // continuation indentée (wrap doux du markdown source)
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^(\s*)([-*]|\d+\.)\s+/.test(lines[i])) {
          item += ' ' + lines[i++].trim();
        }
        items.push(item);
      }
      blocks.push({ t: 'list', ordered, items });
      continue;
    }
    const buf = [line];                                  // paragraphe
    i++;
    while (i < lines.length && lines[i].trim() && !/^(#{1,3}\s|```|\||>|(\s*)([-*]|\d+\.)\s|-{3,}\s*$)/.test(lines[i])) {
      buf.push(lines[i++]);
    }
    blocks.push({ t: 'p', body: buf.join(' ') });
  }
  return blocks;
}

export default function Markdown({ source }) {
  const blocks = parseBlocks(source || '');
  return (
    <div className="flex flex-col gap-3 max-w-none">
      {blocks.map((b, idx) => {
        const key = `b${idx}`;
        switch (b.t) {
          case 'h1': return <h1 key={key} className="m-0 mt-2 text-content font-bold text-2xl">{renderInline(b.body, key)}</h1>;
          case 'h2': return <h2 key={key} className="m-0 mt-4 text-content font-bold text-lg border-b border-border pb-1.5">{renderInline(b.body, key)}</h2>;
          case 'h3': return <h3 key={key} className="m-0 mt-2 text-content font-semibold text-base">{renderInline(b.body, key)}</h3>;
          case 'hr': return <hr key={key} className="border-border my-2" />;
          case 'quote': return (
            <blockquote key={key} className="m-0 rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-4 py-3 text-content text-sm leading-relaxed">
              {renderInline(b.body, key)}
            </blockquote>
          );
          case 'code': return (
            <pre key={key} className="m-0 rounded-lg border border-border bg-app/60 p-3 overflow-x-auto text-[0.8125rem] text-content-muted font-mono">{b.body}</pre>
          );
          case 'table': return (
            <div key={key} className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-surface-raised">
                    {b.header.map((c, ci) => (
                      <th key={ci} className="text-left px-3 py-2 text-content font-semibold border-b border-border whitespace-nowrap">{renderInline(c, `${key}h${ci}`)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {b.body.map((row, ri) => (
                    <tr key={ri} className={ri % 2 ? 'bg-surface' : ''}>
                      {row.map((c, ci) => (
                        <td key={ci} className="px-3 py-2 text-content-muted align-top border-b border-border last:border-b-0">{renderInline(c, `${key}r${ri}c${ci}`)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
          case 'list': {
            const Tag = b.ordered ? 'ol' : 'ul';
            return (
              <Tag key={key} className={`m-0 pl-5 flex flex-col gap-1.5 text-sm text-content-muted ${b.ordered ? 'list-decimal' : 'list-disc'}`}>
                {b.items.map((it, ii) => {
                  const task = it.match(/^\[([ xX])\]\s+(.*)$/);
                  if (task) {
                    return (
                      <li key={ii} className="list-none -ml-5 flex items-start gap-2">
                        <span aria-hidden className={`mt-0.5 grid place-items-center w-4 h-4 shrink-0 rounded border text-[0.625rem] ${task[1] === ' ' ? 'border-border-strong text-transparent' : 'border-emerald-400/60 bg-emerald-500/15 text-emerald-300'}`}>✓</span>
                        <span>{renderInline(task[2], `${key}i${ii}`)}</span>
                      </li>
                    );
                  }
                  return <li key={ii}>{renderInline(it, `${key}i${ii}`)}</li>;
                })}
              </Tag>
            );
          }
          default: return <p key={key} className="m-0 text-sm text-content-muted leading-relaxed">{renderInline(b.body, key)}</p>;
        }
      })}
    </div>
  );
}
