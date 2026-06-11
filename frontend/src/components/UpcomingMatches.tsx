import type { MatchPrediction, Team } from "../types";
import { getUpcomingMatches } from "../utils";
import { MatchCard } from "./MatchCard";

interface UpcomingMatchesProps {
  matches: MatchPrediction[];
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}

export function UpcomingMatches({ matches, teams, onOpen }: UpcomingMatchesProps) {
  const upcoming = getUpcomingMatches(matches);

  return (
    <section className="section section--upcoming" id="upcoming" aria-labelledby="upcoming-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Next up</span>
          <h2 id="upcoming-title">Upcoming matches</h2>
        </div>
        <p>Kickoff times automatically use your local timezone.</p>
      </div>
      {upcoming.length > 0 ? (
        <div className="match-grid">
          {upcoming.map((match) => (
            <MatchCard key={match.id} match={match} teams={teams} onOpen={onOpen} />
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <strong>No upcoming group fixtures remain.</strong>
          <span>Browse the group-stage archive and its original predictions below.</span>
        </div>
      )}
    </section>
  );
}
