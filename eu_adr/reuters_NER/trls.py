# Loosely adapted from:
# http://blog.chrisgreenough.com/2011/01/ws-security-and-python-a-blackboard-story/

import time
import random
import hashlib
import binascii
import logging
from suds.sax.element import Element
from suds.sax.attribute import Attribute

class WSSEDigest(object):
    '''
    classdocs
    '''
    OASIS_PREFIX = "http://docs.oasis-open.org/wss/2004/01/oasis-200401"
    SEC_NS = OASIS_PREFIX + "-wss-wssecurity-secext-1.0.xsd"
    UTIL_NS = OASIS_PREFIX + "-wss-wssecurity-utility-1.0.xsd"
    PASSWORD_DIGEST_TYPE = OASIS_PREFIX + "-wss-username-token-profile-1.0#PasswordDigest"
    NONCE_DIGEST_TYPE = OASIS_PREFIX + "-wss-soap-message-security-1.0#Base64Binary"

    def __init__(self,user,password):
        '''
        Constructor
        '''
        self._user=user
        self._passwordRaw=password
 
    def sign(self):
        # Create the random nonce every time
        m = hashlib.md5()
        m.update(str(random.random()))
        self._nonce=m.hexdigest()
        # define the time
        self._created=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(time.time()))
        # Hash the password
        digest=hashlib.new("sha1",self._nonce+self._created+self._passwordRaw).digest()
        self._password=binascii.b2a_base64(digest)[:-1]


        # create Security element
        securityElem = Element("Security",ns=("wsse", self.SEC_NS))
        securityElem.append(Attribute("SOAP-ENV:mustunderstand", "true"))

        # create UsernameToken elements
        usernameTokenElem = Element("UsernameToken",ns=("wsse", self.SEC_NS))
 
        #create Children of UsernameToken
        createdElem = Element("Created",ns=("wsu",self.UTIL_NS)).setText(self._created)
        usernameElem = Element("Username",ns=("wsse",self.SEC_NS)).setText(self._user)
        #password has already been fully processed
        passwordElem = Element("Password",ns=("wsse",self.SEC_NS)).setText(self._password).append(Attribute("Type",self.PASSWORD_DIGEST_TYPE))
        #base64 the nonce
        nonceElem = Element("Nonce",ns=("wsse",self.SEC_NS)).setText(binascii.b2a_base64(self._nonce)[:-1]).append(Attribute("EncodingType",self.NONCE_DIGEST_TYPE))
 
        # put it all together
        usernameTokenElem.append(usernameElem)
        usernameTokenElem.append(passwordElem)
        usernameTokenElem.append(nonceElem)
        usernameTokenElem.append(createdElem)
        securityElem.append(usernameTokenElem)
        return securityElem
