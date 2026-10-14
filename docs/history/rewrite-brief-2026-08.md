# Celestrium: Rewrite the Astronomy Research Instrument

You are working on **Celestrium**, an open-ended astronomy research tool.

The existing repository has accumulated too much architecture around relatively simple operations. Your job is **not** to preserve the existing architecture. Your job is to preserve the useful scientific capabilities while making the implementation substantially simpler, more direct, inspectable, testable, and extensible.

Think like a very experienced systems programmer who has just inherited a clever but over-engineered codebase.

The guiding principle is:

> **Do the simplest thing that makes the scientific workflow correct.**

Do not add abstractions because they sound architecturally clean. Add them only when they remove real duplication, enforce an important invariant, or make a genuinely difficult boundary simpler.

---

# 1. What Celestrium actually is

Celestrium is a **research instrument**, not an astronomy framework.

It should let a researcher move naturally between:

```text
question
  ↓
literature
  ↓
hypothesis / claim
  ↓
object / region / population
  ↓
catalogue or archive
  ↓
query
  ↓
cross-match / filtering / measurement
  ↓
visualisation
  ↓
statistical test
  ↓
result
  ↓
provenance
```

The researcher should also be able to start anywhere in that loop.

For example:

```text
"M87"
coordinates
a paper
a catalogue
an ADQL query
a Gaia source
a transient
a FITS image
a spectrum
a list of candidates
a proposed physical relationship
```

and then explore outward.

Do **not** constrain the tool to a fixed set of astronomy use cases.

It must remain useful for:

- observational astronomy
- astrophysics
- cosmology
- catalogue research
- archival research
- time-domain astronomy
- exoplanets
- solar-system work
- stellar astronomy
- galaxies and AGN
- high-energy astronomy
- transient astronomy
- spectroscopy
- imaging
- cross-identification
- literature research
- statistical exploration
- hypothesis testing
- exploratory computational work
- reproducible research
- eventually simulation/model comparison where appropriate

The breadth belongs in the **functions**, not in an enormous abstraction hierarchy.

---

# 2. The primary architectural rule

There should be one question behind every design decision:

> **What concrete problem does this abstraction solve?**

If the answer is vague, delete it.

Do not create abstractions merely because:

- two functions look similar;
- something might become a plugin someday;
- a future TUI might need it;
- "separation of concerns" sounds good;
- a design pattern suggests it;
- a registry seems elegant;
- a capability model feels extensible;
- an interface would make the architecture "cleaner".

The code should not contain an elaborate description of itself.

The implementation is the specification.

---

# 3. Prefer functions over frameworks

The core API should be boring.

Something like:

```python
target = resolve("M87")

papers = search_papers("cosmic dipole")

table = query_gaia(...)

matches = xmatch(table, ...)

image = fetch_image(target, ...)

spectrum = fetch_spectrum(target, ...)

plot(table, ...)

result = analyse(...)

save_result(result, ...)
```

The exact API is up to you.

The point is that a researcher should be able to read the code and immediately understand what is happening.

Do not turn:

```python
fetch_spectrum(target)
```

into:

```text
request
→ planner
→ capability
→ observation plan
→ product
→ executor
→ adapter
→ result packet
→ presenter
```

unless the problem genuinely requires that machinery.

---

# 4. Keep the scientific boundaries, not artificial software boundaries

There are real boundaries in this project.

These are worth preserving:

### Network / archive access

Astronomy services are external systems.

Keep their ugly details isolated.

For example:

```text
Gaia
SIMBAD
VizieR
MAST
NED
SDSS
HEASARC
JPL
SkyView
HiPS
ALeRCE
etc.
```

An archive adapter should be thin.

It should mostly translate:

```text
our request
    ↓
archive request
    ↓
archive response
    ↓
useful Python/Astropy object
```

Do not build an abstract "Universal Archive Protocol" unless the actual code demonstrates that one is necessary.

---

# 5. Astronomy objects should use real scientific types

Prefer established scientific representations over homemade ontology.

Use things like:

- `astropy.coordinates.SkyCoord`
- `astropy.table.Table`
- `astropy.units.Quantity`
- `astropy.time.Time`
- FITS / WCS objects
- NumPy arrays
- Pandas where genuinely useful
- standard statistical/scientific libraries

Do not invent a custom representation for something Astropy already represents correctly.

For example, prefer:

```python
SkyCoord(...)
```

to:

```python
ResolvedCoordinate(
    ra=...,
    dec=...,
    frame=...,
    confidence=...,
)
```

unless the additional information is genuinely required.

Likewise, a catalogue row should remain a catalogue row.

Do not wrap every piece of data in a custom object merely to make the architecture "typed".

---

# 6. Target resolution should be simple and honest

Celestrium needs to handle:

```text
M87
NGC 4258
Sirius
Sun
RA DEC
coordinates with frames
catalogue identifiers
aliases
possibly moving objects
```

Resolution should produce enough information to perform useful work.

It should also expose uncertainty or ambiguity when ambiguity exists.

Do not hide uncertainty behind an apparently authoritative `ResolvedTarget`.

A bad match is worse than no match.

If:

```text
M87 → exact
```

say so.

If:

```text
foo → three plausible SIMBAD results
```

say so.

If:

```text
coordinates → field
```

say so.

The user should never have to guess whether Celestrium silently substituted a nearby object.

---

# 7. Do not confuse planning with execution

Planning is useful when the researcher actually needs to decide:

```text
what data exists?
which archive has it?
what is available?
what will this cost?
what is the next useful action?
```

But planning should not become a mandatory ceremony for every operation.

This is fine:

```python
plan("M87")
```

when the user explicitly wants an overview.

This is also fine:

```python
fetch_spectrum("M87")
```

without requiring:

```text
resolve
→ capabilities
→ plan
→ select product
→ execute
```

first.

The convenience API can internally perform the obvious steps.

The implementation should not force the user through the architecture.

---

# 8. No duplicate registries unless absolutely necessary

Be especially suspicious of structures like:

```python
CAPABILITIES = {...}

EXECUTORS = {...}
```

where both dictionaries describe the same conceptual objects.

If metadata and execution need to coexist, find the simplest representation that avoids two sources of truth.

But do not replace two dictionaries with:

```python
MegaProductRegistryFactoryManager(...)
```

That is not simplification.

Sometimes two simple dictionaries are better than one complicated abstraction.

The important rule is:

> **There must be one obvious source of truth for anything that must stay synchronized.**

---

# 9. Archive modules should be boring

A module such as:

```text
gaia.py
mast.py
ned.py
vizier.py
sdss.py
heasarc.py
jpl.py
```

should preferably contain straightforward functions.

For example:

```python
def cone_search(...):
    ...

def fetch_spectrum(...):
    ...

def fetch_photometry(...):
    ...
```

Do not make archive modules implement elaborate interfaces simply because they belong to the same category.

Astronomy archives are not actually identical.

Differences are useful information.

Do not erase them prematurely.

---

# 10. Caching and provenance are first-class

This is one area where architecture is justified.

Scientific reproducibility matters.

If Celestrium performs:

```text
query
download
cross-match
analysis
render
```

the resulting object should be traceable to:

- source/archive
- query
- target coordinates
- relevant parameters
- timestamp where appropriate
- software/version information where appropriate
- source URL or identifier
- cached artifact
- transformation steps where practical

Caching should also be deterministic where possible.

A researcher should be able to answer:

> "Where the hell did this number/image/table come from?"

without reverse-engineering the program.

But do not turn provenance into a distributed ledger.

A plain structured record is probably enough.

---

# 11. Results should be composable

A result should be easy to inspect and pass to another function.

For example:

```python
sources = gaia.cone_search(...)

matches = vizier.cross_match(sources, ...)

plot_matches(matches)
```

or:

```python
spectrum = ned.spectrum("M87")

fit = analyse_spectrum(spectrum)

plot_fit(fit)
```

Avoid a situation where every operation produces a bespoke "result object" that can only be consumed by another layer of Celestrium.

Use normal scientific Python data structures whenever possible.

---

# 12. The CLI is a user interface, not the application

The CLI should be thin.

It should parse arguments and call real functions.

Likewise, the TUI should call those same functions.

Do not put scientific logic inside:

```text
Typer commands
Rich renderers
Textual widgets
```

The CLI should essentially be:

```text
parse
→ call
→ render
```

The TUI should essentially be:

```text
event
→ call
→ render
```

The service functions should not know or care whether they were called from a terminal, TUI, notebook, test, or future web interface.

---

# 13. JSON output matters

Machine-readable output should be straightforward.

If:

```bash
celestrium resolve M87 --json
```

is supported, it should return actual structured data.

Do not make JSON mode a second implementation.

There should be:

```text
one operation
    ↓
different presentation
```

not:

```text
human implementation
+
JSON implementation
```

---

# 14. Keep exploration broad

Do not replace the existing architecture with an artificially tiny application that only does:

```text
resolve
query
image
```

That would miss the point.

The tool should remain capable of growing into a general astronomy research environment.

A researcher should eventually be able to do things such as:

```text
literature search
paper extraction
object resolution
catalogue queries
ADQL
cone searches
cross-matching
spectroscopy
photometry
images
FITS/WCS
time series
light curves
transients
exoplanets
ephemerides
high-energy catalogues
survey coverage
candidate lists
statistical analysis
plots
reports
provenance
reproducible runbooks
```

But each feature should enter the system as a **useful capability**, not as another layer of ontology.

---

# 15. Literature is not a decorative feature

The "theory workshop" side of the project should eventually be able to support a real scientific workflow:

```text
paper
    ↓
claim
    ↓
observable
    ↓
data source
    ↓
test
```

For example:

```text
paper claims X
        ↓
X implies measurable relation Y
        ↓
Y exists in Gaia / SDSS / Euclid / etc.
        ↓
retrieve appropriate sample
        ↓
test Y
        ↓
record result
```

Do not pretend this can be completely automated.

The system should help the researcher move between literature and data without pretending to be an autonomous scientist.

The human remains responsible for interpretation.

---

# 16. Do not hard-code today's science into tomorrow's architecture

The current project may contain specific workflows involving:

- Euclid
- cosmic dipoles
- AGN
- catalogue cross-matching
- transient alerts
- exoplanets
- image reconstruction
- literature claims
- cosmological tests

These are examples.

Do not make them the ontology of the entire system.

If a new researcher wants to investigate:

```text
gravitational lensing
```

or:

```text
stellar streams
```

or:

```text
radio galaxies
```

or:

```text
solar flares
```

or something nobody has thought of yet,

the system should not require architectural surgery.

The extension point should simply be:

> write/use the function that gets the relevant data.

---

# 17. Tests should test behaviour

Do not write tests whose primary purpose is proving that the architecture has been instantiated correctly.

Test things like:

```text
M87 resolves correctly
coordinates remain in the correct frame
ambiguous identifiers remain ambiguous
Gaia query produces expected columns
cache returns identical data
failed network request produces useful error
cross-match respects radius
provenance records source query
JSON output is valid
```

Prefer small tests with obvious failure modes.

Network-dependent tests should be isolated.

Core scientific transformations should be testable offline.

---

# 18. Errors should be boring

A failed archive request should produce:

```text
what failed
where it failed
why it failed if known
what the user can do
```

Do not swallow errors because the UI wants to remain pretty.

Do not turn every exception into a generic:

```text
Something went wrong.
```

Scientific software needs honest failure.

An empty catalogue and a failed request are different things.

An ambiguous target and an unresolved target are different things.

A missing survey footprint and a server outage are different things.

Preserve those distinctions.

---

# 19. Delete dead machinery

During the rewrite, actively look for:

- wrappers around one function
- pass-through classes
- duplicate registries
- unused interfaces
- speculative plugin systems
- compatibility layers that are no longer needed
- abstractions with one implementation
- functions whose only job is calling another function
- models duplicating Astropy objects
- presentation logic leaking into scientific code
- scientific logic hidden inside presentation code
- configuration that merely restates Python constants
- "future-proofing" with no current consumer

Do not preserve something merely because it already exists.

The rewrite is allowed to delete things.

In fact, deletion is one of the main goals.

---

# 20. Keep the useful weirdness

Do not over-sanitize the project into a generic enterprise Python application.

Celestrium is supposed to be an exploratory scientific instrument.

It should remain pleasant to use for someone who wants to ask:

```text
"What's around here?"
```

or:

```text
"What does the literature say about this?"
```

or:

```text
"Show me the optical field."
```

or:

```text
"Which catalogues contain this object?"
```

or:

```text
"Let's test this claim against actual data."
```

That exploratory character is valuable.

The goal is not bureaucratic simplicity.

The goal is **low-friction scientific curiosity**.

---

# 21. The architecture should emerge from usage

Do not design the final architecture first.

Start by identifying the most important workflows.

Implement them directly.

Then look for repeated pain.

Only then extract an abstraction.

The order should be:

```text
working code
    ↓
repeated problem
    ↓
small abstraction
```

Never:

```text
abstract architecture
    ↓
hope future code fits it
```

---

# 22. Definition of done

The rewrite is successful if a competent astronomer/programmer can clone the repository and understand the main workflow quickly.

They should be able to answer:

1. How do I resolve an object?
2. How do I query an archive?
3. How do I retrieve an image?
4. How do I retrieve a spectrum?
5. How do I cross-match catalogues?
6. Where is caching implemented?
7. Where is provenance recorded?
8. How do I add support for another archive?
9. How do I run the CLI?
10. How do I test the scientific logic?

without reading a dissertation about Celestrium's architecture.

If adding a new archive requires understanding twelve layers of infrastructure, the architecture has failed.

If adding a new scientific workflow requires rewriting the architecture, the architecture has failed.

If deleting a module makes the remaining code easier to understand and nothing important breaks, that module probably shouldn't have existed.

---

# 23. Final rule

The system should be:

**small at the core, broad at the edges.**

The core should contain only the genuinely universal things:

```text
scientific data
coordinates
queries
network access
caching
provenance
analysis
```

Everything else should be allowed to remain simple and local.

Do not build a cathedral around a telescope.

Build a good telescope.

Then let scientists point the fucking thing at whatever they want.