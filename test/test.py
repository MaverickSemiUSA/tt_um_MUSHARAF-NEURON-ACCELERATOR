# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, Timer


# Register addresses from the participant RTL
W0   = 0
W1   = 1
W2   = 2
W3   = 3
BIAS = 4
IN0  = 5
IN1  = 6
IN2  = 7
IN3  = 8


def signed6(value):
    """Convert an integer to its 6-bit two's-complement representation."""
    return value & 0x3F


def expected_neuron(weights, inputs, bias):
    """Calculate expected ReLU + saturated 8-bit neuron output."""
    total = sum(w * x for w, x in zip(weights, inputs)) + bias

    if total < 0:
        return 0
    if total > 255:
        return 255
    return total


async def write_reg(dut, address, value):
    """Write a signed 6-bit value into one neuron register."""

    await FallingEdge(dut.clk)

    # uio_in[3:0] = address
    # uio_in[4]   = write enable
    # uio_in[5]   = start
    dut.ui_in.value = signed6(value)
    dut.uio_in.value = (address & 0x0F) | (1 << 4)

    # Register write occurs on this rising edge.
    await RisingEdge(dut.clk)

    # Allow the sequential logic to update before changing controls.
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0


async def run_neuron(dut):
    """Start the neuron and wait for DONE."""

    # Assert START.
    await FallingEdge(dut.clk)
    dut.uio_in.value = 1 << 5

    # START is sampled here.
    await RisingEdge(dut.clk)

    # Wait until the clocked state update has propagated.
    await FallingEdge(dut.clk)

    # Remove START.
    dut.uio_in.value = 0

    # BUSY should now be asserted.
    assert (int(dut.uio_out.value) & 0x01) != 0, (
        "BUSY should be high while the neuron is processing"
    )

    # DONE should still be low.
    assert (int(dut.uio_out.value) & 0x02) == 0, (
        "DONE should be low while the neuron is processing"
    )

    # Wait for DONE.
    for _ in range(20):
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)

        if int(dut.uio_out.value) & 0x02:
            return int(dut.uo_out.value)

    raise AssertionError("Neuron did not assert DONE within 20 clock cycles")


async def setup_neuron(dut, weights, inputs, bias):
    """Load all weights, inputs and bias."""

    await write_reg(dut, W0, weights[0])
    await write_reg(dut, W1, weights[1])
    await write_reg(dut, W2, weights[2])
    await write_reg(dut, W3, weights[3])

    await write_reg(dut, BIAS, bias)

    await write_reg(dut, IN0, inputs[0])
    await write_reg(dut, IN1, inputs[1])
    await write_reg(dut, IN2, inputs[2])
    await write_reg(dut, IN3, inputs[3])


async def check_neuron(dut, weights, inputs, bias):
    """Load a test vector, run the neuron and check the result."""

    expected = expected_neuron(weights, inputs, bias)

    await setup_neuron(dut, weights, inputs, bias)

    result = await run_neuron(dut)

    dut._log.info(
        f"weights={weights}, inputs={inputs}, bias={bias}, "
        f"expected={expected}, result={result}"
    )

    assert result == expected, (
        f"Neuron result mismatch: "
        f"weights={weights}, inputs={inputs}, bias={bias}, "
        f"expected={expected}, got={result}"
    )

    # DONE must be asserted when the result is available.
    assert (int(dut.uio_out.value) & 0x02) != 0, (
        "DONE should be high when result is available"
    )


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start neuron testbench")

    # Clock is generated in tb.v.
    # This avoids Cocotb 2.x Clock period/divisibility issues.
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    dut._log.info("Reset")

    dut.rst_n.value = 0

    for _ in range(5):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 1

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)

    # ------------------------------------------------------------------
    # Check output-enable configuration
    # ------------------------------------------------------------------
    # Participant RTL:
    #   uio_oe[1:0] = 2'b11
    #   uio_oe[7:2] = 6'b0
    assert int(dut.uio_oe.value) == 0x03, (
        f"Unexpected uio_oe value: "
        f"expected 0x03, got 0x{int(dut.uio_oe.value):02X}"
    )

    # ------------------------------------------------------------------
    # Test 1
    #
    # 1*1 + 2*1 + 3*1 + 4*1 + 0 = 10
    # ------------------------------------------------------------------
    dut._log.info("Test 1: basic positive calculation")

    await check_neuron(
        dut,
        weights=[1, 2, 3, 4],
        inputs=[1, 1, 1, 1],
        bias=0,
    )

    # ------------------------------------------------------------------
    # Test 2
    #
    # (-5)*1 + (-5)*1 + (-5)*1 + (-5)*1 + 0 = -20
    # ReLU -> 0
    # ------------------------------------------------------------------
    dut._log.info("Test 2: negative result / ReLU")

    await check_neuron(
        dut,
        weights=[-5, -5, -5, -5],
        inputs=[1, 1, 1, 1],
        bias=0,
    )

    # ------------------------------------------------------------------
    # Test 3
    #
    # 31*4 + 31*4 + 31*4 + 31*4 + 31 = 527
    # Saturation -> 255
    # ------------------------------------------------------------------
    dut._log.info("Test 3: positive saturation")

    await check_neuron(
        dut,
        weights=[31, 31, 31, 31],
        inputs=[4, 4, 4, 4],
        bias=31,
    )

    # ------------------------------------------------------------------
    # Test 4
    #
    # 3*10 + (-2)*5 + 4*2 + (-1)*3 - 5 = 20
    # ------------------------------------------------------------------
    dut._log.info("Test 4: mixed signed values")

    await check_neuron(
        dut,
        weights=[3, -2, 4, -1],
        inputs=[10, 5, 2, 3],
        bias=-5,
    )

    # ------------------------------------------------------------------
    # Test 5
    #
    # (-32)*1 + 0 + 0 + 0 + 0 = -32
    # ReLU -> 0
    # ------------------------------------------------------------------
    dut._log.info("Test 5: minimum signed value")

    await check_neuron(
        dut,
        weights=[-32, 0, 0, 0],
        inputs=[1, 0, 0, 0],
        bias=0,
    )

    # ------------------------------------------------------------------
    # Test 6
    #
    # Check BUSY/DONE timing.
    # ------------------------------------------------------------------
    dut._log.info("Test 6: BUSY/DONE timing")

    await setup_neuron(
        dut,
        weights=[1, 2, 3, 4],
        inputs=[1, 1, 1, 1],
        bias=0,
    )

    # Assert START.
    await FallingEdge(dut.clk)
    dut.uio_in.value = 1 << 5

    # START is sampled here.
    await RisingEdge(dut.clk)

    # Wait for the state transition to propagate.
    await FallingEdge(dut.clk)

    # Remove START.
    dut.uio_in.value = 0

    # BUSY must be high while processing.
    assert (int(dut.uio_out.value) & 0x01) != 0, (
        f"BUSY should be high while processing, "
        f"uio_out=0x{int(dut.uio_out.value):02X}"
    )

    # DONE must not be high yet.
    assert (int(dut.uio_out.value) & 0x02) == 0, (
        f"DONE should be low while processing, "
        f"uio_out=0x{int(dut.uio_out.value):02X}"
    )

    # Wait until DONE.
    for _ in range(20):
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)

        if int(dut.uio_out.value) & 0x02:
            break
    else:
        raise AssertionError("DONE was never asserted")

    # At DONE:
    #   BUSY = 0
    #   DONE = 1
    status = int(dut.uio_out.value)

    assert (status & 0x01) == 0, (
        f"BUSY should be low in DONE state, "
        f"uio_out=0x{status:02X}"
    )

    assert (status & 0x02) != 0, (
        f"DONE should be high in DONE state, "
        f"uio_out=0x{status:02X}"
    )

    # Result should be 10.
    assert int(dut.uo_out.value) == 10, (
        f"Expected result 10, got {int(dut.uo_out.value)}"
    )

    # Wait one more cycle. START is low, so DONE should return to IDLE.
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)

    assert (int(dut.uio_out.value) & 0x02) == 0, (
        "DONE should return low after START is released"
    )

    # IDLE means BUSY should also be low.
    assert (int(dut.uio_out.value) & 0x01) == 0, (
        "BUSY should be low in IDLE state"
    )

    dut._log.info("All neuron tests passed!")
