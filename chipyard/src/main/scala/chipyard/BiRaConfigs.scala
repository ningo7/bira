package chipyard

import org.chipsalliance.cde.config.Config

/** Single Rocket core with BIRA on CUSTOM_3 and the standard 64-bit system bus. */
class BiRaRocketConfig extends Config(
  new bira.WithBiRaRoCC ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig
)
