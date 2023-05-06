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
                const row = document.createElement("tr");

                const emailCell = document.createElement("td");
                emailCell.textContent = email;

                const emailThreat = document.createElement("button");
                emailThreat.innerHTML = 'See threat reason';
                emailThreat.onclick = () => {openThreat(file)};
                emailThreat.classList.add('threatButton');
                emailCell.appendChild(emailThreat);

                row.appendChild(emailCell);

                const dateCell = document.createElement("td");
                dateCell.textContent = date;
                row.appendChild(dateCell);

                row.addEventListener('click', () => {
                    displayFileContent(file);
                });

                inboxElement.appendChild(row);
            });
        });
}

function displayFileContent(filename) {
  fetch(`http://localhost:3001/api/files/${filename}`)
    .then(response => response.text())
    .then(content => {
      const contentElement = document.getElementById('fileContent');
      contentElement.innerHTML = content;

      const lines = contentElement.innerHTML.split('\n');
      lines.splice(0, 16);
      contentElement.innerHTML = lines.join('\n');

      const iframe = contentElement.querySelector('iframe');
      iframe.style.width = '100%';
      iframe.style.height = '100%';
    });
}

displayInbox();
