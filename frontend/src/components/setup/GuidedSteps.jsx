import CopyCommand from './CopyCommand'

export default function GuidedSteps({ intro, steps = [], link, children }) {
  return (
    <div className="space-y-3 text-sm text-content-muted">
      {intro && <p>{intro}</p>}
      <ol className="list-decimal space-y-2 pl-5">
        {steps.map((s, i) => (
          <li key={i} className="space-y-1">
            <span>{s.text}</span>
            {s.command && <CopyCommand command={s.command} />}
          </li>
        ))}
      </ol>
      {link && (
        <a href={link.href} target="_blank" rel="noreferrer"
          className="inline-block text-xs text-primary underline">
          {link.label}
        </a>
      )}
      {children}
    </div>
  )
}
