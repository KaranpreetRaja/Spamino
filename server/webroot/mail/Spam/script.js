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

function displayInbox() {
    const inboxElement = document.getElementById("inbox");

    fetch('http://localhost:3001/api/files')
        .then(response => {
            return response.json();
        })
        .then(files => {
            files.forEach(file => {
                const { email, date } = parseFileName(file);

                mailList = document.getElementById('mailList');
                mailList.innerHTML +=   `<div class="text-white font-bold bg-zinc-700 flex flex-column items-center p-2 rounded-sm mb-4 hover:bg-zinc-800 ease-in-out duration-150">` +
                                        `<button class="rounded-sm bg-red-600 p-1 mr-5 text-white" onclick="openThreat('${file}'); event.stopPropagation();">See threat reason</button>` +
                                        `<p class="cursor-pointer text-xl w-3/4 h-full" onclick="displayFileContent('${file}')">${email}</p>` +
                                        `<p class="ml-auto mr-0 text-neutral-400">${date}</p>` +
                                        `</div>`;
            });
        });
}

function displayFileContent(filename) {
  fetch(`http://localhost:3001/api/files/${filename}`)
    .then(response => response.text())
    .then(content => {
        const { email, date } = parseFileName(filename);

        openPopup(content, email)
    });
}

function openThreat(filename) {
    filename = filename.replace(/mail$/, 'reason');

    fetch(`http://localhost:3002/api/files/${filename}`)
    .then(response => response.text())
    .then(content => {

        openPopup(content, "Reason why this mail got blocked:")
    });
}

function closePopup(){
    popup = document.getElementById('popup');
    popup.classList.add('hidden')
}

function openPopup(content, header){
    popup = document.getElementById('popup');
    popupHeader = document.getElementById('header');
    popup.classList.remove('hidden')
    popupHeader.innerHTML = header

    const contentElement = document.getElementById('fileContent');
    contentElement.innerHTML = content;

    const lines = contentElement.innerHTML.split('\n');
    lines.splice(0, 16);
    contentElement.innerHTML = lines.join('\n');

    const iframe = contentElement.querySelector('iframe');
}

displayInbox();
