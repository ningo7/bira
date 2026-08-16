    .section .rodata.bira_fixture, "a", @progbits
    .balign 64
    .global bira_bfsrcnn_fixture_input
    .type bira_bfsrcnn_fixture_input, @object
bira_bfsrcnn_fixture_input:
    .incbin "../../build/bfsrcnn/fixture/input.bin"
    .size bira_bfsrcnn_fixture_input, .-bira_bfsrcnn_fixture_input

    .balign 64
    .global bira_bfsrcnn_fixture_expected
    .type bira_bfsrcnn_fixture_expected, @object
bira_bfsrcnn_fixture_expected:
    .incbin "../../build/bfsrcnn/fixture/expected.bin"
    .size bira_bfsrcnn_fixture_expected, .-bira_bfsrcnn_fixture_expected

    .section .note.GNU-stack, "", @progbits
