from pathlib import Path

import savethat

if __name__ == "__main__":
    pkg_dir = Path(__file__).parent
    creds = pkg_dir.parent.parent / "savethat_credentials.toml"
    savethat.run_main(
        "lrp_relations", credential_file=creds if creds.exists() else None
    )
