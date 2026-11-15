export default function LearningIllustration() {
  return (
    <div aria-hidden="true" className="hero-card-stack mx-auto hidden h-[22rem] w-full max-w-[29rem] items-center justify-center lg:flex">
      <div className="hero-stack-card hero-stack-card--back-left">
        <span className="hero-stack-pill">SOURCE NOTES</span><p>Cellular respiration</p><small>Energy, stored and released.</small>
      </div>
      <div className="hero-stack-card hero-stack-card--back-right">
        <span className="hero-stack-pill">UNDERSTANDING</span><p>ATP powers the cell.</p>
      </div>
      <div className="hero-stack-card hero-stack-card--front">
        <span className="hero-stack-pill">QUESTION</span><p>What powers the cell’s work?</p><small>Tap a card. Recall the idea.</small>
      </div>
    </div>
  )
}
