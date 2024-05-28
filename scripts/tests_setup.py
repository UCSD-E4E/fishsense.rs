import json
import os
from pathlib import Path
from typing import Dict

from nas_unzip.nas import nas_unzip


def get_creds() -> Dict[str, str]:
    """Obtains the credentials

    Returns:
        Dict[str, str]: Username and password dictionary
    """
    if Path("credentials.json").is_file():
        with open("credentials.json", "r", encoding="ascii") as handle:
            return json.load(handle)
    else:
        value = os.environ["NAS_CREDS"].splitlines()
        assert len(value) == 2
        return {"username": value[0], "password": value[1]}


def download_data(creds: Dict[str, str]):
    script_path = Path(__file__)

    data_path = script_path.parent.parent / "data"
    data_path.unlink(missing_ok=True)
    data_path.mkdir()

    nas_unzip(
        network_path="smb://e4e-nas.ucsd.edu:6021/temp/github_actions/fishsensers/fishsensersTest.zip",
        output_path= data_path.absolute(),
        username=creds["username"],
        password=creds["password"],
    )


if __name__ == "__main__":
    creds = get_creds()
    download_data(creds)
