#! /usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import os

    import build

    from driver.main import main

    config = build.load_build_configs(os.path.dirname(__file__))
    main(release_build=config.pop("DEFAULT"), debug_build=config.pop("DEBUG"))
