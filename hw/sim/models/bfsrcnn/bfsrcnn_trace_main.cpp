#include "VBiRaStandaloneTop.h"
#include "verilated.h"

// BFSRCNN-specific trace replay and stage checking. The Verilated
// BiRaStandaloneTop and its row-level memory protocol remain model-agnostic.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint8_t kFenceFunct = 0x47;
constexpr uint8_t kStoreFunct = 0x46;

struct Command {
  uint8_t funct;
  bool returns_value;
  uint64_t rs1;
  uint64_t rs2;
};

template <typename T>
T read_scalar(std::istream &stream) {
  T value{};
  stream.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!stream) {
    throw std::runtime_error("unexpected end of binary file");
  }
  return value;
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + path);
  }
  stream.seekg(0, std::ios::end);
  const auto size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char *>(data.data()), size);
  if (!stream) {
    throw std::runtime_error("cannot read " + path);
  }
  return data;
}

struct Fixture {
  uint64_t memory_base;
  std::vector<Command> commands;
  std::vector<uint8_t> memory;
  std::vector<uint8_t> stages;
  std::vector<uint8_t> expected;
  size_t input_pixels;
};

Fixture load_fixture(const std::string &root) {
  std::ifstream stream(root + "/commands.bin", std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open commands.bin");
  }
  std::array<char, 8> magic{};
  stream.read(magic.data(), magic.size());
  if (std::string(magic.data(), magic.size()) != "BIRA_CMD") {
    throw std::runtime_error("invalid commands.bin magic");
  }
  if (read_scalar<uint32_t>(stream) != 1u) {
    throw std::runtime_error("unsupported commands.bin version");
  }
  const uint32_t count = read_scalar<uint32_t>(stream);
  Fixture fixture{};
  fixture.memory_base = read_scalar<uint64_t>(stream);
  fixture.commands.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    Command command{};
    command.funct = read_scalar<uint8_t>(stream);
    command.returns_value = (read_scalar<uint8_t>(stream) & 1u) != 0u;
    std::array<char, 6> padding{};
    stream.read(padding.data(), padding.size());
    command.rs1 = read_scalar<uint64_t>(stream);
    command.rs2 = read_scalar<uint64_t>(stream);
    fixture.commands.push_back(command);
  }
  fixture.memory = read_file(root + "/memory.bin");
  fixture.stages = read_file(root + "/stages.bin");
  fixture.expected = read_file(root + "/expected.bin");
  fixture.input_pixels = read_file(root + "/input.bin").size();
  return fixture;
}

class Simulation {
 public:
  explicit Simulation(Fixture fixture)
      : fixture_(std::move(fixture)), context_(new VerilatedContext),
        dut_(new VBiRaStandaloneTop(context_.get())) {
    max_cycles_ =
        std::max<uint64_t>(10000000u, fixture_.input_pixels * 100000u);
    dut_->clock = 0;
    dut_->reset = 0;
    dut_->io_command_valid = 0;
    dut_->io_response_ready = 1;
    dut_->io_dramReadResponse_valid = 0;
    dut_->io_dramWriteResponse_valid = 0;
    dut_->io_dramReadRequest_ready = 0;
    dut_->io_dramWriteRequest_ready = 0;
    dut_->eval();
  }

  ~Simulation() {
    dut_->final();
  }

  void run() {
    reset();
    const std::vector<std::string> layers = {
        "head",     "shrink1",  "shrink2",  "shrink3", "mapping0",
        "mapping1", "mapping2", "mapping3", "mapping4", "mapping5",
        "mapping6", "mapping7", "expand",   "final",
    };
    uint64_t phase_start = cycles_;
    size_t fence_index = 0;

    for (const auto &command : fixture_.commands) {
      issue(command);
      if (command.returns_value) {
        while (responses_.empty()) {
          tick();
        }
        const uint64_t result = responses_.front();
        responses_.erase(responses_.begin());
        if (command.funct == kFenceFunct) {
          if ((result & 1u) == 0u || ((result >> 1u) & 0xffu) != 0u) {
            throw std::runtime_error("FENCE returned an error");
          }
          const uint64_t elapsed = cycles_ - phase_start;
          if (fence_index < 2u * (layers.size() - 1u)) {
            const auto &layer = layers.at(fence_index / 2u);
            if ((fence_index & 1u) == 0u) {
              layer_cycles_.emplace_back(layer, elapsed);
            } else {
              debug_cycles_.emplace_back(layer, elapsed);
            }
          } else {
            layer_cycles_.emplace_back(layers.back(), elapsed);
          }
          ++fence_index;
          phase_start = cycles_;
        }
      }
    }
    if (fence_index != 2u * layers.size() - 1u) {
      throw std::runtime_error("unexpected FENCE count");
    }
    verify_outputs();
    if (dut_->io_busy) {
      throw std::runtime_error("accelerator remained busy after trace");
    }
    report();
  }

 private:
  uint8_t memory_read(uint64_t address) const {
    if (address < fixture_.memory_base) {
      throw std::runtime_error("DRAM read below memory base");
    }
    const uint64_t offset = address - fixture_.memory_base;
    if (offset >= fixture_.memory.size()) {
      throw std::runtime_error("DRAM read outside memory image");
    }
    return fixture_.memory.at(static_cast<size_t>(offset));
  }

  void memory_write(uint64_t address, uint8_t value) {
    if (address < fixture_.memory_base) {
      throw std::runtime_error("DRAM write below memory base");
    }
    const uint64_t offset = address - fixture_.memory_base;
    if (offset >= fixture_.memory.size()) {
      throw std::runtime_error("DRAM write outside memory image");
    }
    fixture_.memory.at(static_cast<size_t>(offset)) = value;
  }

  void set_read_data(const std::array<uint32_t, 16> &words) {
    for (size_t index = 0; index < words.size(); ++index) {
      dut_->io_dramReadResponse_bits_data[index] = words[index];
    }
  }

  bool tick() {
    if (cycles_ >= max_cycles_) {
      throw std::runtime_error("simulation exceeded maximum cycle count");
    }
    dut_->clock = 0;
    dut_->io_dramReadRequest_ready = !read_pending_;
    dut_->io_dramWriteRequest_ready = !write_pending_;
    dut_->io_dramReadResponse_valid = read_pending_;
    set_read_data(read_data_);
    dut_->io_dramReadResponse_bits_errorCode = 0;
    dut_->io_dramWriteResponse_valid = write_pending_;
    dut_->io_dramWriteResponse_bits_errorCode = 0;
    dut_->eval();

    const bool command_fire =
        dut_->io_command_valid && dut_->io_command_ready;
    const bool read_request_fire =
        dut_->io_dramReadRequest_valid && dut_->io_dramReadRequest_ready;
    const bool read_response_fire =
        dut_->io_dramReadResponse_valid && dut_->io_dramReadResponse_ready;
    const bool write_request_fire =
        dut_->io_dramWriteRequest_valid && dut_->io_dramWriteRequest_ready;
    const bool write_response_fire =
        dut_->io_dramWriteResponse_valid && dut_->io_dramWriteResponse_ready;
    const bool response_fire =
        dut_->io_response_valid && dut_->io_response_ready;

    std::array<uint32_t, 16> next_read{};
    if (read_request_fire) {
      const uint64_t address =
          dut_->io_dramReadRequest_bits_virtualAddress;
      const unsigned count = dut_->io_dramReadRequest_bits_bytes;
      for (unsigned byte = 0; byte < count; ++byte) {
        next_read[byte / 4u] |=
            static_cast<uint32_t>(memory_read(address + byte))
            << (8u * (byte % 4u));
      }
    }
    if (write_request_fire) {
      const uint64_t address =
          dut_->io_dramWriteRequest_bits_virtualAddress;
      const unsigned count = dut_->io_dramWriteRequest_bits_bytes;
      for (unsigned byte = 0; byte < count; ++byte) {
        const uint32_t word =
            dut_->io_dramWriteRequest_bits_data[byte / 4u];
        memory_write(
            address + byte,
            static_cast<uint8_t>(word >> (8u * (byte % 4u))));
      }
    }
    if (response_fire) {
      responses_.push_back(dut_->io_response_bits_data);
    }

    dut_->clock = 1;
    dut_->eval();
    context_->timeInc(1);
    ++cycles_;

    if (read_response_fire) {
      read_pending_ = false;
    }
    if (read_request_fire) {
      read_data_ = next_read;
      read_pending_ = true;
    }
    if (write_response_fire) {
      write_pending_ = false;
    }
    if (write_request_fire) {
      write_pending_ = true;
    }
    return command_fire;
  }

  void reset() {
    dut_->reset = 1;
    for (unsigned count = 0; count < 3; ++count) {
      tick();
    }
    dut_->reset = 0;
    tick();
  }

  void issue(const Command &command) {
    dut_->io_command_bits_funct = command.funct;
    dut_->io_command_bits_rs1 = command.rs1;
    dut_->io_command_bits_rs2 = command.rs2;
    dut_->io_command_bits_rd = 1;
    dut_->io_command_bits_xd = command.returns_value;
    dut_->io_command_bits_xs1 = 1;
    dut_->io_command_bits_xs2 = 1;
    dut_->io_command_bits_translationStatus = 0;
    dut_->io_command_valid = 1;
    bool accepted = false;
    while (!accepted) {
      accepted = tick();
    }
    dut_->io_command_valid = 0;
  }

  void verify_range(
      uint64_t address,
      const std::vector<uint8_t> &expected,
      size_t expected_offset,
      size_t size,
      const std::string &name) const {
    for (size_t index = 0; index < size; ++index) {
      const auto actual = memory_read(address + index);
      const auto reference = expected.at(expected_offset + index);
      if (actual != reference) {
        throw std::runtime_error(
            name + " mismatch at byte " + std::to_string(index));
      }
    }
  }

  void verify_outputs() const {
    std::vector<const Command *> stores;
    for (const auto &command : fixture_.commands) {
      if (command.funct == kStoreFunct) {
        stores.push_back(&command);
      }
    }
    const size_t pixels = fixture_.input_pixels;
    std::vector<size_t> sizes = {
        pixels * 48u, pixels * 32u, pixels * 32u,
    };
    sizes.insert(sizes.end(), 9u, pixels * 16u);
    sizes.push_back(pixels * 128u);
    if (stores.size() != sizes.size() + 1u) {
      throw std::runtime_error("unexpected STORE count");
    }
    size_t offset = 0;
    for (size_t stage = 0; stage < sizes.size(); ++stage) {
      verify_range(
          stores.at(stage)->rs1,
          fixture_.stages,
          offset,
          sizes.at(stage),
          "stage " + std::to_string(stage));
      offset += sizes.at(stage);
    }
    if (offset != fixture_.stages.size()) {
      throw std::runtime_error("unexpected stages.bin size");
    }
    verify_range(
        stores.back()->rs1,
        fixture_.expected,
        0,
        fixture_.expected.size(),
        "final output");
  }

  void report() const {
    uint64_t compute_total = 0;
    uint64_t debug_total = 0;
    std::cout << "input_pixels=" << fixture_.input_pixels << '\n';
    for (const auto &[name, cycles] : layer_cycles_) {
      std::cout << "layer." << name << ".cycles=" << cycles << '\n';
      compute_total += cycles;
    }
    for (const auto &[name, cycles] : debug_cycles_) {
      std::cout << "debug_store." << name << ".cycles=" << cycles << '\n';
      debug_total += cycles;
    }
    std::cout << "profiled_layer_cycles=" << compute_total << '\n';
    std::cout << "debug_store_cycles=" << debug_total << '\n';
    std::cout << "total_cycles=" << cycles_ << '\n';
  }

  Fixture fixture_;
  std::unique_ptr<VerilatedContext> context_;
  std::unique_ptr<VBiRaStandaloneTop> dut_;
  uint64_t cycles_ = 0;
  uint64_t max_cycles_ = 0;
  bool read_pending_ = false;
  bool write_pending_ = false;
  std::array<uint32_t, 16> read_data_{};
  std::vector<uint64_t> responses_;
  std::vector<std::pair<std::string, uint64_t>> layer_cycles_;
  std::vector<std::pair<std::string, uint64_t>> debug_cycles_;
};

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: " << argv[0] << " FIXTURE_DIR\n";
      return 2;
    }
    Simulation simulation(load_fixture(argv[1]));
    simulation.run();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
