"""`python -m celestrium.tui` → launch the Celestrium cockpit."""
import sys

if __name__ == "__main__":
    if any(arg in ("--help", "-h") for arg in sys.argv[1:]):
        # A bare module launch must behave like a normal Python entry point,
        # not silently start a full-screen app that then has to be Ctrl+C'd.
        print("Celestrium cockpit — a Textual TUI, no CLI flags of its own.\n"
              "Usage: python -m celestrium.tui\n"
              "       python -m celestrium tui   (the documented front door)")
        sys.exit(0)
    from .app import main
    main()
