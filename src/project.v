// =============================================================================
// Design 3 (clean): Single Neuron / Perceptron Unit
// Fully unsigned datapath - no `signed` reg/wire declarations, no signed
// comparisons anywhere. Two's-complement values are stored as plain unsigned
// bit patterns and manually sign-extended. Only ONE localized $signed() cast
// exists (the multiply itself), which is standard and synthesis-safe.
// This avoids known signed-comparator mismapping issues between behavioral
// sim and gate-level synthesis on some Yosys/PDK toolchains.
// =============================================================================
`default_nettype none

module tt_um_neuron (
    input  wire [7:0] ui_in,    // data_in[7:0]: byte written into selected register
                                 // (only lower 6 bits [5:0] are stored)
    output wire [7:0] uo_out,   // neuron output (ReLU + saturated, unsigned 8-bit)
    input  wire [7:0] uio_in,   // [3:0]=addr, [4]=wr_en, [5]=start, [7:6]=unused
    output wire [7:0] uio_out,  // [0]=busy, [1]=done, [7:2]=0
    output wire [7:0] uio_oe,   // [1:0]=output, [7:2]=input
    input  wire        ena,
    input  wire        clk,
    input  wire        rst_n
);

    // -------------------------------------------------------------------
    // Control field mapping
    // -------------------------------------------------------------------
    wire [5:0] data_in = ui_in[5:0]; // narrowed to 6 bits at the write port
    wire [3:0] addr    = uio_in[3:0];
    wire       wr_en   = uio_in[4];
    wire       start   = uio_in[5];

    // -------------------------------------------------------------------
    // Register file: 4 weights, 1 bias, 4 inputs. Plain unsigned storage
    // holding two's-complement bit patterns (range represents -32..31).
    // Address map: 0-3=w0..w3, 4=bias, 5-8=in0..in3
    // -------------------------------------------------------------------
    reg [5:0] w0, w1, w2, w3;
    reg [5:0] bias;
    reg [5:0] in0, in1, in2, in3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w0   <= 6'd0; w1 <= 6'd0; w2 <= 6'd0; w3 <= 6'd0;
            bias <= 6'd0;
            in0  <= 6'd0; in1 <= 6'd0; in2 <= 6'd0; in3 <= 6'd0;
        end else if (ena && wr_en) begin
            case (addr)
                4'd0: w0   <= data_in;
                4'd1: w1   <= data_in;
                4'd2: w2   <= data_in;
                4'd3: w3   <= data_in;
                4'd4: bias <= data_in;
                4'd5: in0  <= data_in;
                4'd6: in1  <= data_in;
                4'd7: in2  <= data_in;
                4'd8: in3  <= data_in;
                default: ; // no-op for unused addresses
            endcase
        end
    end

    // -------------------------------------------------------------------
    // FSM: time-multiplexed MAC over 4 weight*input products, add bias,
    // then apply ReLU + saturation.
    // -------------------------------------------------------------------
    localparam [2:0] S_IDLE = 3'd0,
                      S_MAC0 = 3'd1,
                      S_MAC1 = 3'd2,
                      S_MAC2 = 3'd3,
                      S_MAC3 = 3'd4,
                      S_BIAS = 3'd5,
                      S_ACT  = 3'd6,
                      S_DONE = 3'd7;

    reg [2:0]  state;
    reg [15:0] acc;      // unsigned storage of a two's-complement accumulator value
    reg [7:0]  result;

    // Shared 6x6 multiplier input mux (plain unsigned storage of raw bit patterns)
    reg [5:0] mult_a, mult_b;
    always @* begin
        case (state)
            S_MAC0: begin mult_a = w0; mult_b = in0; end
            S_MAC1: begin mult_a = w1; mult_b = in1; end
            S_MAC2: begin mult_a = w2; mult_b = in2; end
            S_MAC3: begin mult_a = w3; mult_b = in3; end
            default: begin mult_a = 6'd0; mult_b = 6'd0; end
        endcase
    end

    // Single localized signed cast for the multiply itself - necessary and
    // standard; result is immediately captured back into a plain unsigned wire.
    wire [11:0] mult_out = $signed(mult_a) * $signed(mult_b); // 6x6 -> 12-bit

    // Manual sign-extension to 16 bits (no signed arithmetic involved)
    wire [15:0] mult_ext = {{4{mult_out[11]}}, mult_out};
    wire [15:0] bias_ext = {{10{bias[5]}}, bias};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            acc    <= 16'd0;
            result <= 8'd0;
        end else if (ena) begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        acc   <= 16'd0;
                        state <= S_MAC0;
                    end
                end
                S_MAC0: begin acc <= acc + mult_ext; state <= S_MAC1; end
                S_MAC1: begin acc <= acc + mult_ext; state <= S_MAC2; end
                S_MAC2: begin acc <= acc + mult_ext; state <= S_MAC3; end
                S_MAC3: begin acc <= acc + mult_ext; state <= S_BIAS; end
                S_BIAS: begin acc <= acc + bias_ext; state <= S_ACT;  end
                S_ACT: begin
                    // Pure bit-test for sign, plain unsigned compare for
                    // saturation - no signed comparators anywhere.
                    if (acc[15])
                        result <= 8'd0;             // negative -> ReLU clamps to 0
                    else if (acc > 16'd255)
                        result <= 8'd255;            // positive overflow -> saturate
                    else
                        result <= acc[7:0];
                    state <= S_DONE;
                end
                S_DONE: begin
                    if (!start) state <= S_IDLE;     // wait for start to deassert
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    wire busy = (state != S_IDLE) && (state != S_DONE);
    wire done = (state == S_DONE);

    // -------------------------------------------------------------------
    // Output decode
    // -------------------------------------------------------------------
    assign uo_out  = result;
    assign uio_out = {6'b0, done, busy};
    assign uio_oe  = 8'b0000_0011; // [1:0]=output (busy,done), [7:2]=input

    // Avoid unused-signal lint warnings for reserved control/data bits
    wire _unused_ok = &{uio_in[7:6], ui_in[7:6], 1'b0};

endmodule

`default_nettype wire
