plugins {
    kotlin("jvm") version "1.9.23"
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "io.ailake"
version = "0.1.6"

repositories {
    mavenCentral()
}

val trinoVersion = "430"

dependencies {
    // Trino SPI — provided at runtime by the Trino server
    compileOnly("io.trino:trino-spi:$trinoVersion")
    compileOnly("io.airlift:slice:2.2")
    compileOnly("com.fasterxml.jackson.core:jackson-annotations:2.15.2")

    // SLF4J — Trino's isolated per-plugin classloader does NOT expose slf4j-api to
    // plugins (confirmed live: compileOnly here produced NoClassDefFoundError:
    // org/slf4j/LoggerFactory at connector construction, VectorScanMetadata.<init>,
    // on a real Trino 460 server — unlike trino-spi/airlift-slice, which the
    // classloader does share). Must be bundled in the shadowJar.
    implementation("org.slf4j:slf4j-api:2.0.9")

    // JNA — bundled in the plugin fat-jar
    implementation("net.java.dev.jna:jna:5.14.0")

    // Jackson for JSON parsing of native results
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.15.2")

    testImplementation(kotlin("test"))
    testImplementation("io.trino:trino-spi:$trinoVersion")
    testImplementation("io.airlift:slice:2.2")
    testImplementation("org.mockito.kotlin:mockito-kotlin:5.2.1")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.1")
    testRuntimeOnly("org.slf4j:slf4j-simple:2.0.9")
}

tasks.shadowJar {
    archiveClassifier.set("plugin")
    // Exclude Trino SPI and its transitive deps (provided by Trino server)
    dependencies {
        exclude(dependency("io.trino:.*"))
        exclude(dependency("io.airlift:slice:.*"))
    }
    mergeServiceFiles()
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(17)
}
