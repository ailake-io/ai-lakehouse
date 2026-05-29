// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
package io.ailake.flink

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class AilakeVectorConnectorFactoryTest {

    @Test
    fun factoryIdentifier() {
        assertEquals("ailake", AilakeVectorConnectorFactory().factoryIdentifier())
    }

    @Test
    fun catalogFactoryIdentifier() {
        assertEquals("ailake", AilakeCatalogFactory().factoryIdentifier())
    }

    @Test
    fun requiredOptions() {
        val factory = AilakeVectorConnectorFactory()
        val keys = factory.requiredOptions().map { it.key() }
        assert("warehouse" in keys)
        assert("table-name" in keys)
        assert("vector.dim" in keys)
    }
}
