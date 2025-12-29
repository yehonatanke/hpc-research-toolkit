import datetime as dt


COLOR = {
    'blue': '\033[94m',
    'default': '\033[99m',
    'grey': '\033[90m',
    'yellow': '\033[93m',
    'black': '\033[90m',
    'cyan': '\033[96m',
    'green': '\033[92m',
    'magenta': '\033[95m',
    'white': '\033[97m',
    'red': '\033[91m'
}

def print_color_palette():
    print("\n=== ANSI Color Palette ===\n")

    # Foreground colors (38;2;r;g;b)
    print("Truecolor (RGB) samples:\n")

    colors = [
        ("Soft Blue", "130;170;255"),
        ("Teal/Green", "120;220;180"),
        ("Warm Orange", "255;200;100"),
        ("Soft Red", "255;120;130"),
        ("Bright Red", "255;80;100"),
        ("Light Purple", "180;140;255"),
        ("Muted Gray", "140;140;140"),
        ("Gray-Blue", "100;110;130"),
        ("Cyan", "80;200;220"),
        ("Lime", "180;255;100"),
        ("Pink", "255;140;180"),
        ("Amber", "255;180;80"),
        ("Violet", "160;120;255"),
        ("Dark Gray", "90;90;90"),
        ("Light Gray", "200;200;200"),
    ]

    for name, rgb in colors:
        print(f"\x1b[38;2;{rgb}m● {name.ljust(15)} \x1b[38;2;{rgb}m{name}\x1b[0m")

    print("\nCommon 16-color alternatives:\n")
    basic = [
        ("Black", "30"),
        ("Red", "31"),
        ("Green", "32"),
        ("Yellow", "33"),
        ("Blue", "34"),
        ("Magenta", "35"),
        ("Cyan", "36"),
        ("White", "37"),
        ("Bright Black", "90"),
        ("Bright Red", "91"),
        ("Bright Green", "92"),
        ("Bright Yellow", "93"),
        ("Bright Blue", "94"),
        ("Bright Magenta", "95"),
        ("Bright Cyan", "96"),
        ("Bright White", "97"),
    ]

    for name, code in basic:
        print(f"\x1b[{code}m● {name.ljust(15)} \x1b[{code}mSample Text\x1b[0m", end="  ")
        if (basic.index((name, code)) + 1) % 4 == 0:
            print()

    print("\n\nSuggested logging color schemes:\n")

    schemes = {
        "Modern Soft": {
            "TIMESTAMP": "100;110;130",
            "DEBUG": "130;170;255",
            "INFO": "120;220;180",
            "WARNING": "255;200;100",
            "ERROR": "255;120;130",
            "CRITICAL": "255;80;100",
            "LOCATION": "180;140;255",
            "EXTRA": "140;140;140",
            "EXCEPTION": "255;100;100",
        },
        "Vivid": {
            "TIMESTAMP": "90;90;90",
            "DEBUG": "100;200;255",
            "INFO": "100;255;150",
            "WARNING": "255;220;100",
            "ERROR": "255;100;100",
            "CRITICAL": "255;50;80",
            "LOCATION": "200;150;255",
            "EXTRA": "180;180;180",
            "EXCEPTION": "255;80;80",
        },
        "Pastel": {
            "TIMESTAMP": "120;120;140",
            "DEBUG": "150;180;255",
            "INFO": "150;220;200",
            "WARNING": "255;220;150",
            "ERROR": "255;160;160",
            "CRITICAL": "255;120;140",
            "LOCATION": "200;170;255",
            "EXTRA": "160;160;160",
            "EXCEPTION": "255;130;130",
        },
    }

    for scheme_name, colors in schemes.items():
        print(f"{scheme_name}:")
        for role, rgb in colors.items():
            print(f"  \x1b[38;2;{rgb}m● {role.ljust(10)} \x1b[38;2;{rgb}m{rgb}\x1b[0m")
        print()


if __name__ == "__main__":
    print_color_palette()
