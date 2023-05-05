document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("login-form");
    const usernameInput = document.getElementById("username");
    const passwordInput = document.getElementById("password");

    loginForm.addEventListener("submit", (e) => {
        e.preventDefault();

        const username = usernameInput.value;
        const password = passwordInput.value;

        if (username === "spamino@testmail.local" && password === "1234") {
            window.location.href = "/mail";
        } else {
            alert("Invalid username or password.");
        }
    });
});
