plugins {
    scala
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "io.ailake"
version = "0.1.1"

repositories {
    mavenCentral()
}

val sparkVersion = "3.5.0"
val scalaVersion = "2.12"

dependencies {
    // Spark — provided by the cluster at runtime
    compileOnly("org.apache.spark:spark-sql_$scalaVersion:$sparkVersion")
    compileOnly("org.apache.spark:spark-catalyst_$scalaVersion:$sparkVersion")
    compileOnly("org.scala-lang:scala-library:2.12.18")

    // JNA — bundled in the plugin jar (Spark does not provide it)
    implementation("net.java.dev.jna:jna:5.14.0")

    // Jackson — provided by Spark at runtime; compileOnly avoids bundling it.
    // Using direct import instead of Class.forName to fail fast if unavailable.
    compileOnly("com.fasterxml.jackson.core:jackson-databind:2.15.2")

    testImplementation("org.apache.spark:spark-sql_$scalaVersion:$sparkVersion")
    testImplementation("org.scalatest:scalatest_$scalaVersion:3.2.17")
    testImplementation("org.scalatestplus:junit-4-13_$scalaVersion:3.2.17.0")
}

tasks.shadowJar {
    archiveClassifier.set("plugin")
    dependencies {
        exclude(dependency("org.apache.spark:.*"))
        exclude(dependency("org.scala-lang:.*"))
    }
    mergeServiceFiles()
}

tasks.test {
    useJUnit()
    // Spark 3.x accesses JDK internals sealed in JDK 17+
    jvmArgs(
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    )
}
