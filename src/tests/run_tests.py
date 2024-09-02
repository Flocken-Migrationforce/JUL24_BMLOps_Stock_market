import pytest
import sys
import os

def run_tests():
    # Setze das aktuelle Verzeichnis auf den Ordner, in dem sich dieses Skript befindet
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Führe pytest aus und erhalte das Ergebnis
    result = pytest.main(["-q", "test_app.py"])

    # Überprüfe das Ergebnis
    if result == 0:
        print("All tests passed successfully.")
    else:
        print(f"{result} test(s) failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()

