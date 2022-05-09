import coverage
import savethat

# ensures coverage is also collected for subprocesses
coverage.process_startup()

savethat.run_main("sanity_checks_for_relation_networks")
