plugins {
    kotlin("jvm") version "1.9.23"
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "io.ailake"
version = "0.1.1"

repositories {
    mavenCentral()
}

val flinkVersion = "1.18.1"
val scalaVersion = "2.12"

dependencies {
    // Flink Table API — provided at runtime by the Flink cluster
    compileOnly("org.apache.flink:flink-table-api-java:$flinkVersion")
    compileOnly("org.apache.flink:flink-table-common:$flinkVersion")
    compileOnly("org.apache.flink:flink-streaming-java:$flinkVersion")
    compileOnly("org.apache.flink:flink-table-api-java-bridge:$flinkVersion")
    compileOnly("org.apache.flink:flink-java:$flinkVersion")

    // JNA — bundled in the fat-jar (Flink does not provide it)
    implementation("net.java.dev.jna:jna:5.14.0")

    // Jackson for JSON parsing of native results
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")

    testImplementation(kotlin("test"))
    testImplementation("org.apache.flink:flink-table-api-java:$flinkVersion")
    testImplementation("org.apache.flink:flink-table-common:$flinkVersion")
    testImplementation("org.apache.flink:flink-streaming-java:$flinkVersion")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.1")
    testImplementation("org.mockito.kotlin:mockito-kotlin:5.2.1")
}

tasks.shadowJar {
    archiveClassifier.set("plugin")
    dependencies {
        exclude(dependency("org.apache.flink:.*"))
    }
    mergeServiceFiles()
}

tasks.test {
    useJUnitPlatform()
    // Forward ailake.native.lib system property to the test JVM so that
    // `gradle test -Dailake.native.lib=/path/to/libailake_jni.so` works.
    // CI uses AILAKE_NATIVE_LIB env var (inherited automatically).
    System.getProperty("ailake.native.lib")?.let { systemProperty("ailake.native.lib", it) }
}

kotlin {
    jvmToolchain(17)
}
