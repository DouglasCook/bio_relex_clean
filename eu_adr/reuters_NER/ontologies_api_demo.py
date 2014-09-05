"""
Provided by Thomson Reuters, tweaked by me
"""

import ConfigParser
from suds.client import Client
from suds import WebFault
from suds.bindings.binding import Binding
from trls import WSSEDigest

import xml.etree.ElementTree as etreeXML
import re

config = ConfigParser.ConfigParser()
config.read('api_login.cfg')
# TODO ideally get config file working
#user = config.get('api', 'user')
#password = config.get('api', 'password')
user = 'Imperial001'
password = 'ZAFPILE0BJGND4W8'

'''
def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)
'''

def summariseXml(apiOutput):
    matches = {}

    if hasattr(apiOutput, 'Entities'):

        try:
            for i in apiOutput.Entities.Entity:
                # need to cast as strings here so they can be used later
                e_type = str(i._type)
                start = i._start
                end = i._end
                entity = str(i._originalForm)

                # don't want P < 0.001 recognised as a drug, brackets and capitalisation mean need to use find
                if entity.find('.001') == -1:
                    matches[entity] = [e_type, start, end]
        except:
            pass

    return matches


def runOntologiesSearch(testPhrase):
    Binding.replyfilter = (lambda s, r: ''.join(r[r.find('<soap:Envelope'):r.rfind('Envelope>') + 9]))
    wsse = WSSEDigest(user, password)

    endpointUrl = 'https://lsapi.thomson-pharma.com/ls-api-ws/ws/LSApiService/ontologies/v1'
    wsdlUrl = 'https://lsapi.thomson-pharma.com/ls-api-ws/ws/LSApiService/ontologies/v1?wsdl'

    client = Client(wsdlUrl)
    client.location = endpointUrl
    client.set_options(soapheaders=wsse.sign())

    # NER SEARCH TO GET MATCHES - Note this is the wsdl type, not name, so starts with lowercase letters
    namedEntityRecognitionInput = client.factory.create('namedEntityRecognitionInput')
    namedEntityRecognitionInput.text = testPhrase
    # TODO some problem with setting the format here, is it necessary?
    #namedEntityRecognitionInput.fmt = 'xml'

    try:
        output = client.service.searchNer(namedEntityRecognitionInput)
    except WebFault, e:
        error = 'get synonyms error ', e

    return output


if __name__ == '__main__':
    #output = runOntologiesSearch(
        #'sd pfizer gilenya acetominophen ache injection assay phase 1 il8 ace sildenafil or acetaminophen with a phase(2) trial progesterone')
    # summariseXml(output)

    output = runOntologiesSearch('Multiple regression analysis showed that pancreas-to-muscle SI ratios on T1-weighted images and ADC values were independently associated with pancreatic fibrosis (r(2) = 0.66, P < .001) and with activated PSC expression (r(2) = 0.67, P < .001).')
    print summariseXml(output)
'''
filename = 'C:/' 	# e.g. 'C:/tmp/text.txt'
f = open(filename, w)
print output
f.write(output)
f.close
'''