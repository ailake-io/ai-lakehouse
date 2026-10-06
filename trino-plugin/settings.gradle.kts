dependencyResolutionManagement {
    versionCatalogs {
        create("ailakeLibs") {
            from(files("../gradle/ailake-plugins.versions.toml"))
        }
    }
}

rootProject.name = "trino-plugin"
