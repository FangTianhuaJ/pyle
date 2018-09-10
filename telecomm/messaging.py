import smtplib

SMS_GATEWAYS = {
                'AMIT':'4088936758@txt.att.net',
                'DANIEL':'3018012855@messaging.sprintpcs.com',
                'TED':'6077270880@txt.att.net',
                'YU':'6122271557@messaging.sprintpcs.com',
                'PETER':'5102066192@txt.att.net',
                'JULIAN':'2068199718@vzwpix.com',
                'CHARLES':'8053366385@mymetropcs.com'}

def textMessage(recipients, subject, msg, username='LabRAD',attempts=2):
    """Send a text message to one or more recipients

    INPUTS:
    recipients - str or [str,str...]: List of names of labmembers
    to whom you want to send the message. These names must be in the
    SMS_GATEWAYS dictionary.
    subject - str: Subject of the message
    msg - str: message to send
    username - str, optional: PCS account name to stick on the message. Defaults to 'LabRAD'
    """
    if not isinstance(recipients,list):
        recipients = [recipients]
    try:
        email([SMS_GATEWAYS[recip.upper()] for recip in recipients], subject, msg, username,attempts=attempts)
    except:
        print 'Text message failed'
        raise

def email(toAddrs,subject,msg,username='LabRAD',attempts=2,noisy=False):
    """Send an email to one or more recipients

    INPUTS:
    toAddrs - str or [str...]: target address or list of target addresses
    subject - str: Subject of the message
    msg - str: message to send
    username - str, optional: PCS account name to stick on the message. Defaults to 'LabRAD'
    """
    fromAddr = username+'@physics.ucsb.edu'
    if not isinstance(toAddrs,list):
        toAddrs = [toAddrs]
    header = """From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n"""%(fromAddr,", ".join(toAddrs), subject)
    message = header+msg
    if noisy:
        print 'Sending message:\r\n-------------------------\r\n'+message+'\r\n-------------------------\r\n'
        print '\n'
    for attempt in range(attempts):
        try:
            print 'Sending message from %s' %fromAddr
            server = smtplib.SMTP('smtp.physics.ucsb.edu')
            server.sendmail(fromAddr, toAddrs, message)
            print 'Message sent'
            server.quit()
            break
        except:
            print 'Error. Message not sent'
            if attempt<attempts-1:
                print 'Trying again. This is attempt %d' %(attempt+1)
                continue
            else:
                print 'Maximum retries reached'
                raise
                