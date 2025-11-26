def test_linecount():
    """
    the test first needs to run `black` and `isort` function on the entire files in backend/*.py
    and then count the number of the lines.
    the test should pass if the number of the lines is less than 200 in total.
    """

    # DO NOT CHANGE THE CODE BELOW. THIS IS A CRITICAL TEST.
    import os
    import subprocess

    subprocess.run(["black", "backend"])
    subprocess.run(["isort", "backend"])
    num_lines = 0
    for file in os.listdir("backend"):
        if file.endswith(".py"):
            with open(os.path.join("backend", file), "r") as f:
                num_lines += len(f.readlines())
    print(f"Number of lines: {num_lines}")

    assert num_lines < 1000


def test_flake8():
    import glob
    import subprocess

    files = glob.glob("backend/*.py")

    result = subprocess.run(["flake8"] + files, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)

    assert result.returncode == 0
