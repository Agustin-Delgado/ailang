# toposort

Standard input describes build steps. Each non-empty line is `STEP: DEP1 DEP2 ...` meaning STEP depends
on the listed steps (the list may be empty: `STEP:`). A step that only appears as a dependency is a step
with no dependencies. Print an order in which every step can run after all of its dependencies; when
several steps are ready at the same time, take the alphabetically smallest first. Print the steps
separated by single spaces on one line. If the dependencies contain a cycle, print `cycle` instead.
