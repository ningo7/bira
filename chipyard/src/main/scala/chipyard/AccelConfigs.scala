package chipyard

import org.chipsalliance.cde.config.Config

/** Single Rocket core with BIRA on CUSTOM_3 and a 128-bit DMA/system bus. */
class AccelRocketConfig extends Config(
  new bira.WithRoCCAccel ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)
