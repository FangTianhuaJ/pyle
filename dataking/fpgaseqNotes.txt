#Current
runQubits
 p = server.packet()
 makeSequence(p, devices)
  Add readout resonator sram pulses
  Find envelope times
  p.initialize([(d.__name__, d['channels']) for d in devices])
  addConfig(p, devices)
  p.new_mem()
  addMem(p, ...)
  addSram(p, ...)
 p.build_sequence()
 p.run(long(stats))
 p.<get data>
 return sendPacket(p, debug)


#Transmon
runQubits(devices)
 p = server.packet()
 makeSequence(p, devices)
  1. #Find envelope times
  2. #initialize
     p.initialize
     addConfig(p, devices)
  3. #Memory sequence
     p.new_mem()
     addMem()
      Go to operation bias for as setting time
      Run SRAM
      Go to zero    ###GOOD TO HERE
      ???Delay long enough for the readout and data sendback to finish 
  4. #SRAM
     addSram(p)
      Figure out x,y,z data, whatever.
      Add the readout pulses however they're current implemented, but
      taking the start delay into account.
  5. #Finish up and send packet
  p.build_sequence()
  p.run(long(stats))
  p.<get data>
  return 
