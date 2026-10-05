/*
    JavaScript Scope & Hoisting Practice

    Concepts covered:
    - Global scope
    - Function scope
    - Block scope
    - let vs const vs var
    - Variable shadowing
    - Hoisting
    - Temporal Dead Zone
*/


// ==========================================
// 1. GLOBAL SCOPE
// ==========================================

let appName = "Web Development";

function showAppName() {
    console.log("\nGlobal variable:");
    console.log(appName);
}

showAppName();


// ==========================================
// 2. FUNCTION SCOPE
// ==========================================

function userDetails() {
    let username = "Shyam";

    console.log("\nInside function:");
    console.log(username);
}

userDetails();

// This would cause an error because username
// exists only inside userDetails().

// console.log(username);


// ==========================================
// 3. BLOCK SCOPE
// ==========================================

if (true) {
    let message = "Inside the block";

    console.log("\nBlock scope:");
    console.log(message);
}

// message cannot be accessed here.

// console.log(message);


// ==========================================
// 4. let IS BLOCK-SCOPED
// ==========================================

if (true) {
    let number = 100;
    console.log("\nlet inside block:", number);
}

// console.log(number); // Error


// ==========================================
// 5. const IS ALSO BLOCK-SCOPED
// ==========================================

if (true) {
    const country = "India";

    console.log("\nconst inside block:", country);
}

// console.log(country); // Error


// ==========================================
// 6. var IS FUNCTION-SCOPED
// ==========================================

function testVar() {

    if (true) {
        var value = "JavaScript";
    }

    // var is accessible outside the block
    // because it is function-scoped.

    console.log("\nvar inside function:", value);
}

testVar();


// ==========================================
// 7. VARIABLE SHADOWING
// ==========================================

let name = "Global Shyam";

function showName() {

    let name = "Local Shyam";

    console.log("\nInside function:");
    console.log(name);
}

showName();

console.log("Outside function:");
console.log(name);


// ==========================================
// 8. HOISTING WITH var
// ==========================================

console.log("\nHoisting with var:");

console.log(age);

var age = 22;


// JavaScript behaves approximately like:
//
// var age;
// console.log(age);
// age = 22;
//
// Result: undefined


// ==========================================
// 9. let AND TEMPORAL DEAD ZONE
// ==========================================

// Uncommenting the following code causes an error.

// console.log(score);
// let score = 100;


// Unlike var, let is not safely accessible
// before its declaration.


// ==========================================
// 10. const BEFORE DECLARATION
// ==========================================

// Also causes an error.

// console.log(role);
// const role = "Developer";


// ==========================================
// 11. FUNCTION HOISTING
// ==========================================

sayHello();

function sayHello() {
    console.log("\nFunction declarations are hoisted.");
}


// ==========================================
// 12. PRACTICAL EXAMPLE
// ==========================================

let user = "Shyam";

function login() {

    let status = "Logged in";

    if (status === "Logged in") {
        let message = `${user} successfully logged in.`;

        console.log("\nLogin:");
        console.log(message);
    }
}

login();
