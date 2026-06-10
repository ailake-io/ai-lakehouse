# SPDX-License-Identifier: MIT OR Apache-2.0
"""Apache Airflow provider for AI-Lake Format."""

__version__ = "0.0.16"


def get_provider_info():
    return {
        "package-name": "apache-airflow-providers-ailake",
        "name": "AI-Lake Format",
        "description": "Hook, operators, and snapshot sensor for AI-Lake tables.",
        "versions": [__version__],
        "connection-types": [
            {
                "connection-type": "ailake",
                "hook-class-name": "airflow_providers_ailake.hooks.ailake.AilakeHook",
            }
        ],
    }
