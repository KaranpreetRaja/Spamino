#!/bin/bash

ssh root@192.168.122.146 'bash -c "python3 /root/Spamino/mainFilter.py"'
spamcheck=$(ls /spams/reason)
spamname=$(basename $spamcheck)

if [ -z $spamname ]; then
   echo "email isn't spam"
   currentmail=$(ls /mails | head)
   returnpath=$(grep "Return-Path" /mails/$currentmail | cut -d ":" -f 2 | tr -d ' ')
   to=$(grep "To:" /mails/$currentmail | cut -d ":" -f 2 | tr -d ' ')
   date=$(date '+%a, %d %b %Y %H:%M:%S')
   date2=$(date '+%a %b %d %H:%M:%S %Y')
   outputdate=$(date '+%Y-%m-%d-%H-%M-%S')
   emailuser="<spamino@testmail.local>"
   adminuser="<admin@testmail.local>"

   sed -i 's/$to/$aemailuser/g' /mails/$currentmail
   sed -E -i "s/([A-Za-z]{3}, [0-9]{2} [A-Za-z]{3} [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2})/${date}/g" /mails/$currentmail
   sed -E -i "s/([A-Za-z]{3} [A-Za-z]{3} [0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4})/${date2}/g" /mails/$currentmail
   sed -E -i "s/<([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>/${emailuser}/g" /mails/$currentmail
   perl -0777 -pe "s/([A-Za-z]{3}, [0-9]{2} [A-Za-z]{3} [0-9]{4}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\s+\+?[0-9]{4})/${date}/g" /mails/$currentmail > /tmp/testmail
   mv /tmp/testmail /mails/$currentmail
   sed -i 's/$returnpath/$adminuser/g' /mails/$currentmail

   from=$(grep "From:" /mails/$currentmail | grep -i -E -o '\b[A-Za-z0-9._%+-]+#?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b')
   modifiedfrom=$(echo "$from" | sed -e 's/@/-/g' -e 's/\$/-/g' -e 's/#/-/g' -e 's/%/-/g' -e 's/\^/-/g' -e 's/!/-/g' -e 's/\*/-/g' -e 's/&/-/g' -e 's/(/-/g' -e 's/)/-/g' -e 's/_/-/g' -e 's/=/-/g' -e 's/+/-/g' -e 's/\./-/g' -e 's/,/-/g' -e 's/\//-/g' -e 's/;/-/g' -e 's/:/-/g' -e "s/'/-/g" -e 's/"/-/g' -e 's/\[/-/g' -e 's/\]/-/g' -e 's/{/-/g' -e 's/}/-/g' -e 's/|/-/g' -e 's/\\\\/-/g')

   mv /mails/$currentmail /var/www/html/mail/Emails/$outputdate-$modifiedfrom.mail
else
   echo "email is spam"
   currentmail=$(ls /spams | grep -v "reason")
   returnpath=$(grep "Return-Path" /spams/$currentmail | cut -d ":" -f 2 | tr -d ' ')
   to=$(grep "To:" /spams/$currentmail | cut -d ":" -f 2 | tr -d ' ')
   date=$(date '+%a, %d %b %Y %H:%M:%S')
   date2=$(date '+%a %b %d %H:%M:%S %Y')
   outputdate=$(date '+%Y-%m-%d-%H-%M-%S')
   emailuser="<spamino@testmail.local>"
   adminuser="<admin@testmail.local>"

   sed -i 's/$to/$emailuser/g' /spams/$currentmail
   sed -E -i "s/([A-Za-z]{3}, [0-9]{2} [A-Za-z]{3} [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2})/${date}/g" /spams/$currentmail
   sed -E -i "s/([A-Za-z]{3} [A-Za-z]{3} [0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4})/${date2}/g" /spams/$currentmail
   sed -E -i "s/<([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>/${emailuser}/g" /spams/$currentmail
   perl -0777 -pe "s/([A-Za-z]{3}, [0-9]{2} [A-Za-z]{3} [0-9]{4}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\s+\+?[0-9]{4})/${date}/g" /spams/$currentmail > /tmp/testmail
   mv /tmp/testmail /spams/$currentmail
   sed -i 's/$returnpath/$adminuser/g' /spams/$currentmail

   from=$(grep "From:" /spams/$currentmail | grep -i -E -o '\b[A-Za-z0-9._%+-]+#?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b')
   modifiedfrom=$(echo "$from" | sed -e 's/@/-/g' -e 's/\$/-/g' -e 's/#/-/g' -e 's/%/-/g' -e 's/\^/-/g' -e 's/!/-/g' -e 's/\*/-/g' -e 's/&/-/g' -e 's/(/-/g' -e 's/)/-/g' -e 's/_/-/g' -e 's/=/-/g' -e 's/+/-/g' -e 's/\./-/g' -e 's/,/-/g' -e 's/\//-/g' -e 's/;/-/g' -e 's/:/-/g' -e "s/'/-/g" -e 's/"/-/g' -e 's/\[/-/g' -e 's/\]/-/g' -e 's/{/-/g' -e 's/}/-/g' -e 's/|/-/g' -e 's/\\\\/-/g')

   mv /spams/$currentmail /var/www/html/mail/Spam/$outputdate-$modifiedfrom.mail
   mv /spams/$spamname /var/www/html/mail/Spam/$outputdate-$modifiedfrom.reason
fi
