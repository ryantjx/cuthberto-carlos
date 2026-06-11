import type { MouseEvent } from "react";
import type { MatchPrediction, Team } from "../types";
import {
  formatKickoffParts,
  formatPercent,
  mostLikelyOutcome,
} from "../utils";
import { TeamFlag } from "./TeamFlag";

interface MatchCardProps {
  match: MatchPrediction;
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}

export function MatchCard({ match, teams, onOpen }: MatchCardProps) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  const probabilities = match.prediction.probabilities;
  const [homeScore, awayScore] = match.prediction.mostLikelyScore;

  function handleOpen(event: MouseEvent<HTMLButtonElement>) {
    onOpen(match, event.currentTarget);
  }

  return (
    <article className="match-card">
      <div className="match-card__meta">
        <span className="eyebrow">Group {match.group}</span>
        <span>{kickoff.date}</span>
        <strong>{kickoff.time}</strong>
      </div>
      <div className="match-card__teams">
        <TeamFlag team={teams[match.homeTeam]} />
        <span className="match-card__score" aria-label="Most likely score">
          {homeScore}–{awayScore}
        </span>
        <TeamFlag team={teams[match.awayTeam]} />
      </div>
      <p className="match-card__venue">{match.venue}</p>
      <div className="probability-strip" aria-label="Result probabilities">
        <span
          className="probability-strip__home"
          style={{ width: `${probabilities.homeWin * 100}%` }}
          title={`${match.homeTeam} ${formatPercent(probabilities.homeWin, 1)}`}
        />
        <span
          className="probability-strip__draw"
          style={{ width: `${probabilities.draw * 100}%` }}
          title={`Draw ${formatPercent(probabilities.draw, 1)}`}
        />
        <span
          className="probability-strip__away"
          style={{ width: `${probabilities.awayWin * 100}%` }}
          title={`${match.awayTeam} ${formatPercent(probabilities.awayWin, 1)}`}
        />
      </div>
      <div className="match-card__probabilities" aria-hidden="true">
        <span>H {formatPercent(probabilities.homeWin)}</span>
        <span>D {formatPercent(probabilities.draw)}</span>
        <span>A {formatPercent(probabilities.awayWin)}</span>
      </div>
      <div className="match-card__footer">
        <span>{mostLikelyOutcome(probabilities, match.homeTeam, match.awayTeam)}</span>
        <span className="match-card__actions">
          <button className="text-button" type="button" onClick={handleOpen}>
            Explore prediction
          </button>
          <a className="text-link" href={match.sourceUrl} target="_blank" rel="noreferrer">
            Source <span aria-hidden="true">↗</span>
          </a>
        </span>
      </div>
    </article>
  );
}
