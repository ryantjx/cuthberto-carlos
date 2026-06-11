import type { MouseEvent } from "react";
import type { MatchPrediction, Team } from "../types";
import { formatKickoffParts, formatPercent, mostLikelyOutcome } from "../utils";
import { TeamFlag } from "./TeamFlag";

interface MatchListRowProps {
  match: MatchPrediction;
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}

export function MatchListRow({ match, teams, onOpen }: MatchListRowProps) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  const probabilities = match.prediction.probabilities;
  const [homeScore, awayScore] = match.prediction.mostLikelyScore;

  function handleOpen(event: MouseEvent<HTMLButtonElement>) {
    onOpen(match, event.currentTarget);
  }

  return (
    <article className="match-list-row">
      <div className="match-list-row__kickoff">
        <span className="eyebrow">Group {match.group}</span>
        <strong>{kickoff.date}</strong>
        <span>{kickoff.time}</span>
      </div>
      <div className="match-list-row__fixture">
        <TeamFlag team={teams[match.homeTeam]} compact />
        <strong className="match-list-row__score" aria-label="Most likely score">
          {homeScore}–{awayScore}
        </strong>
        <TeamFlag team={teams[match.awayTeam]} compact />
      </div>
      <div className="match-list-row__prediction">
        <span>{mostLikelyOutcome(probabilities, match.homeTeam, match.awayTeam)}</span>
        <div aria-label="Result probabilities">
          <span>H {formatPercent(probabilities.homeWin)}</span>
          <span>D {formatPercent(probabilities.draw)}</span>
          <span>A {formatPercent(probabilities.awayWin)}</span>
        </div>
      </div>
      <span className="match-list-row__venue">{match.venue}</span>
      <span className="match-list-row__actions">
        <button
          className="text-button match-list-row__action"
          type="button"
          onClick={handleOpen}
          aria-label={`Explore prediction for ${match.homeTeam} versus ${match.awayTeam}`}
        >
          Explore
        </button>
        <a
          className="text-link"
          href={match.sourceUrl}
          target="_blank"
          rel="noreferrer"
          aria-label={`View source data for ${match.homeTeam} versus ${match.awayTeam}`}
        >
          Source <span aria-hidden="true">↗</span>
        </a>
      </span>
    </article>
  );
}
