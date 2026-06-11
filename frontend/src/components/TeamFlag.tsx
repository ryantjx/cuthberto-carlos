import type { Team } from "../types";
import { flagUrls } from "../flags";

interface TeamFlagProps {
  team: Team;
  compact?: boolean;
}

export function TeamFlag({ team, compact = false }: TeamFlagProps) {
  return (
    <span className={`team-identity${compact ? " team-identity--compact" : ""}`}>
      <img className="team-flag" src={flagUrls[team.flagCode]} alt="" aria-hidden="true" />
      <span>{team.name}</span>
    </span>
  );
}
