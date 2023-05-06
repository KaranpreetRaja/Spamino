currentSelected = -1;

function parseFileName(fileName) {
    fileName = fileName.replace(/\.mail$/, '');
    const dateRegex = /^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-/;
    const dateMatch = fileName.match(dateRegex);
    const date = dateMatch[0].slice(0, 10).replace(/-/g, '/');
    const time = dateMatch[0].slice(11, 19).replace(/-/g, ':');

    const emailParts = fileName.slice(dateMatch[0].length).split('-');
    const domain = emailParts.slice(-2).join('.');
    emailParts.splice(-2, 2);
    const email = emailParts.join('-') + '@' + domain;

    return { email, date: `${date} ${time}` };
}

function parseEmail(mailBody) {
    var obj = {};
    mailBody.split('\n').forEach(v => v.replace(/\s*(.*)\s*:\s*(.*)\s*/, (s, key, val) => {
        obj[key] = isNaN(val) || val.length < 1 ? val || undefined : Number(val);
    }));
    return obj
}

function displayInbox() {

    fetch('http://localhost:3001/api/files')
        .then(response => {
            return response.json();
        })
        .then(files => {
            let i = 0
            files.forEach(file => {
                const { email, date } = parseFileName(file);
                fetch(`http://localhost:3001/api/files/${file}`)
                    .then(response => response.text())
                    .then(content => {
                        let mail = parseEmail(content)
                        console.log(mail);
                        mailList = document.getElementById('inbox');
                        mailList.innerHTML +=
                            `<div id="${i}" class="webix_list_item inboxItem" style="width: auto; height: 112px; overflow: hidden" role="option" aria-selected="true" tabindex="0"  
                            onclick="displayFileContent('${file}', ${i})"> 
                                <div class="inboxWrapp flex">
                                <div class="inboxLeft">
                                </div>
                                <div class="flex flexColumn wide">
                                    <div class="flex spaceBetween wide">
                                    <span class="inboxUser">${email}</span>
                                    <span class="smTextLight">${date}</span>
                                    </div>
                                    <div class="inboxSubject">${mail.subj}</div>
                                    <div class="flex spaceBetween alignCenter wide">
                                    
                                    <span class="inboxTag mdi mdi-tag work"></span>
                                    </div>
                                </div>
                                </div>
                            </div>`;
                            i++;
                    });
                // <span class="inboxMessage">${content}</span>

                `<div class="text-white font-bold bg-zinc-700 flex flex-column items-center p-2 rounded-sm mb-4 hover:bg-zinc-800 ease-in-out duration-150">` +
                    `<button class="rounded-sm bg-red-600 p-1 mr-5 text-white" onclick="openThreat('${file}'); event.stopPropagation();">See threat reason</button>` +
                    `<p class="cursor-pointer text-xl w-3/4 h-full" >${email}</p>` +
                    `<p class="ml-auto mr-0 text-neutral-400">${date}</p>` +
                    `</div>`;

                
            });
        });
}

function displayFileContent(filename, id) {
    fetch(`http://localhost:3001/api/files/${filename}`)
        .then(response => response.text())
        .then(content => {
            const { email, date } = parseFileName(filename);
            let mail = parseEmail(content)

            element = document.getElementById(currentSelected);
            if(element)
                element.classList.remove("webix_selected");

            element = document.getElementById(id);
            element.classList.add("webix_selected");
            currentSelected = id;

            sub = document.getElementById('mailSubject')
            sub.innerHTML = mail.subj

            sender = document.getElementById('sender');
            recipient = document.getElementById('recipient');
            dateShow = document.getElementById('date');

            sender.innerHTML = email
            recipient.innerHTML = mail.To
            dateShow.innerHTML = date


            const contentElement = document.getElementById('fileContent');
            contentElement.innerHTML = content;

            const lines = contentElement.innerHTML.split('\n');
            lines.splice(0, 16);
            contentElement.innerHTML = lines.join('\n');

            const iframe = contentElement.querySelector('iframe');
        });
}

function openThreat(filename) {
    filename = filename.replace(/mail$/, 'reason');

    fetch(`http://localhost:3002/api/files/${filename}`)
        .then(response => response.text())
        .then(content => {
            openPopupReason(content, "Reason why this mail got blocked:")
        });
}

function displayContent(content) {

}

function closePopup() {
    popup = document.getElementById('popup');
    popup.classList.add('hidden')
}

function openPopup(content, header) {
    popup = document.getElementById('popup');
    popupHeader = document.getElementById('header');
    popup.classList.remove('hidden')
    popupHeader.innerHTML = header


}

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        closePopup();
    }
});

function openPopupReason(content, header) {
    const popup = document.getElementById('popup');
    const popupHeader = document.getElementById('header');
    const contentElement = document.getElementById('fileContent');

    popup.classList.remove('hidden');
    popupHeader.innerHTML = header;
    contentElement.textContent = content;
}

displayInbox();
