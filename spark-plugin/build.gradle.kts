plugins {
    scala
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "io.ailake"
version = "0.1.0"

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
}
