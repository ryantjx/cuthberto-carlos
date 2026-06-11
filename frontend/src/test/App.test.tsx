import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "../App";
import tournamentData from "../data/tournament.json";

describe("generated tournament data", () => {
  it("contains the complete initial tournament shape", () => {
    expect(tournamentData.snapshotDate).toBe("2026-06-11");
    expect(tournamentData.groupMatches).toHaveLength(72);
    expect(tournamentData.groups).toHaveLength(12);
    expect(tournamentData.groups.every((group) => group.matchIds.length === 6)).toBe(true);
    expect(tournamentData.knockoutMatches).toHaveLength(32);
  });
});

describe("App interactions", () => {
  it("filters groups and opens an accessible prediction drawer", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /^B$/ }));
    expect(screen.getByRole("heading", { name: "Group B" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Group A" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Open prediction for Canada versus Bosnia and Herzegovina/i }));
    expect(screen.getByRole("dialog", { name: /Canada vs Bosnia and Herzegovina/i })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /Score probability/i })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Close prediction details" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("keeps official knockout feeder labels visible", () => {
    render(<App />);
    expect(screen.getAllByText("2A").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Runner-up Group A").length).toBeGreaterThan(0);
  });

  it("updates knockout details when a bracket match is selected", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /M104 W101 Winner Match 101 W102 Winner Match 102/ }));
    expect(screen.getByLabelText("Details for Match 104")).toHaveTextContent("W101 vs W102");
  });
});
