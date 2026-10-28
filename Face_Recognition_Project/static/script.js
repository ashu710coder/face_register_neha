// ============================================================
// ELEMENTS
// ============================================================

const video =
    document.getElementById("video");


const canvas =
    document.getElementById("canvas");


const faceBox =
    document.getElementById("faceBox");


const faceStatus =
    document.getElementById("faceStatus");


const registerBtn =
    document.getElementById("registerBtn");


const employeeIdInput =
    document.getElementById("employeeId");


const employeeNameInput =
    document.getElementById("employeeName");


const registerStatus =
    document.getElementById("registerStatus");


const progressBar =
    document.getElementById("progressBar");


const progressText =
    document.getElementById("progressText");


const inBtn =
    document.getElementById("inBtn");


const outBtn =
    document.getElementById("outBtn");


const attendanceStatus =
    document.getElementById(
        "attendanceStatus"
    );


const recognizedId =
    document.getElementById(
        "recognizedId"
    );


const recognizedName =
    document.getElementById(
        "recognizedName"
    );


const recognizedScore =
    document.getElementById(
        "recognizedScore"
    );


// ============================================================
// SETTINGS
// ============================================================

const TOTAL_SAMPLES = 25;

let registrationRunning = false;

let faceDetected = false;


// ============================================================
// START CAMERA
// ============================================================

async function startCamera() {

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    width: {
                        ideal: 640
                    },

                    height: {
                        ideal: 480
                    }

                },

                audio: false

            });


        video.srcObject = stream;


        await video.play();


        faceStatus.innerText =
            "Camera ready";


        // Start live detection

        startFaceDetection();


    }

    catch (error) {

        console.error(error);


        faceStatus.innerText =
            "Camera permission denied";


        alert(
            "Camera permission allow karo."
        );

    }

}


// ============================================================
// LIVE FACE DETECTION
// ============================================================

function startFaceDetection() {

    // NOTE:
    // Browser side par native face detection har browser
    // me available nahi hota.
    //
    // Isliye yahan basic camera status hai.
    //
    // Actual OpenCV face detection registration ke time
    // Flask server par hota hai.


    setInterval(
        checkCamera,
        500
    );

}


function checkCamera() {

    if (
        video.videoWidth > 0
    ) {

        faceStatus.innerText =
            "Camera active - position your face in front of the camera";

    }

}


// ============================================================
// CAPTURE FRAME
// ============================================================

function captureFrame() {

    canvas.width =
        video.videoVideoWidth ||
        video.videoWidth;


    canvas.height =
        video.videoHeight;


    const context =
        canvas.getContext("2d");


    context.drawImage(

        video,

        0,

        0,

        canvas.width,

        canvas.height

    );


    return canvas;

}


// ============================================================
// WAIT
// ============================================================

function wait(ms) {

    return new Promise(

        resolve => {

            setTimeout(
                resolve,
                ms
            );

        }

    );

}


// ============================================================
// CANVAS TO BLOB
// ============================================================

function canvasToBlob(canvas) {

    return new Promise(

        resolve => {

            canvas.toBlob(

                resolve,

                "image/jpeg",

                0.95

            );

        }

    );

}


// ============================================================
// REGISTER
// ============================================================

registerBtn.addEventListener(

    "click",

    startRegistration

);


async function startRegistration() {

    if (
        registrationRunning
    ) {

        return;

    }


    const empId =
        employeeIdInput
            .value
            .trim();


    const empName =
        employeeNameInput
            .value
            .trim();


    // Validation

    if (!empId) {

        alert(
            "Employee ID enter karo."
        );

        return;

    }


    if (!empName) {

        alert(
            "Employee Name enter karo."
        );

        return;

    }


    if (
        !video.srcObject
    ) {

        alert(
            "Camera did not start."
        );

        return;

    }


    // Start

    registrationRunning =
        true;


    registerBtn.disabled =
        true;


    inBtn.disabled =
        true;


    outBtn.disabled =
        true;


    employeeIdInput.disabled =
        true;


    employeeNameInput.disabled =
        true;


    registerStatus.innerText =
        "Registration started...";


    let validSamples = 0;


    let attempts = 0;


    // ========================================================
    // IMPORTANT
    //
    // 25 VALID faces capture karne hain.
    // Agar server face detect nahi karta to count nahi badhega.
    // ========================================================

    while (
        validSamples < TOTAL_SAMPLES
    ) {


        attempts++;


        // Safety limit

        if (
            attempts > 100
        ) {

            alert(
                "Face is not being detected properly.. "
                + "Look clearly towards the camera."
            );

            break;

        }


        registerStatus.innerText =

            `Face detect/capture: `
            + `${validSamples}/${TOTAL_SAMPLES}`;


        faceStatus.innerText =
            "Face is being detected...";


        // Wait

        await wait(700);


        // Capture

        const frame =
            captureFrame();


        const blob =
            await canvasToBlob(
                frame
            );


        // Form

        const formData =
            new FormData();


        formData.append(

            "employee_id",

            empId

        );


        formData.append(

            "employee_name",

            empName

        );


        formData.append(

            "image",

            blob,

            `face_${validSamples + 1}.jpg`

        );


        try {

            const response =
                await fetch(

                    "/register",

                    {

                        method:
                            "POST",

                        body:
                            formData

                    }

                );


            const data =
                await response.json();


            console.log(
                data
            );


            // =================================================
            // FACE DETECTED
            // =================================================

            if (
                data.success
            ) {

                validSamples =
                    data.image_number;


                faceDetected =
                    true;


                faceStatus.innerText =
                    "✓ FACE DETECTED";


                faceStatus.style.color =
                    "green";


                registerStatus.innerText =

                    `Face captured `
                    + `${validSamples}/`
                    + `${TOTAL_SAMPLES}`;


                progressBar.style.width =

                    `${(
                        validSamples /
                        TOTAL_SAMPLES
                    ) * 100}%`;


                progressText.innerText =

                    `${validSamples}/`
                    + `${TOTAL_SAMPLES}`;


                // Training complete

                if (
                    data.training_completed
                ) {

                    registerStatus.innerText =
                        "Registration + Training completed!";

                    break;

                }

            }

            // =================================================
            // FACE NOT DETECTED
            // =================================================

            else {

                faceDetected =
                    false;


                faceStatus.innerText =
                    "✗ FACE NOT DETECTED";


                faceStatus.style.color =
                    "red";


                registerStatus.innerText =

                    "Face not detected. "
                    + "Look clearly towards the camera.";

            }


        }

        catch (error) {

            console.error(
                error
            );


            registerStatus.innerText =
                "Server error.";

        }

    }


    // ========================================================
    // FINISH
    // ========================================================

    if (
        validSamples >= TOTAL_SAMPLES
    ) {

        registerStatus.innerText =
            "✓ Face Registration Completed";


        progressBar.style.width =
            "100%";


        progressText.innerText =
            "25/25";


        alert(

            "FACE REGISTRATION SUCCESSFUL!\n\n"

            + "Employee ID: "
            + empId

            + "\nEmployee Name: "
            + empName

            + "\n\n25 face samples captured."

            + "\nModel training completed."

        );

    }


    // Reset

    registrationRunning =
        false;


    registerBtn.disabled =
        false;


    inBtn.disabled =
        false;


    outBtn.disabled =
        false;


    employeeIdInput.disabled =
        false;


    employeeNameInput.disabled =
        false;

}


// ============================================================
// MARK IN
// ============================================================

inBtn.addEventListener(

    "click",

    function () {

        markAttendance(
            "IN"
        );

    }

);


// ============================================================
// MARK OUT
// ============================================================

outBtn.addEventListener(

    "click",

    function () {

        markAttendance(
            "OUT"
        );

    }

);


// ============================================================
// ATTENDANCE
// ============================================================

async function markAttendance(
    mode
) {

    attendanceStatus.innerText =

        `${mode}: Face scanning...`;


    recognizedId.innerText =
        "-";


    recognizedName.innerText =
        "-";


    recognizedScore.innerText =
        "-";


    // Capture

    const frame =
        captureFrame();


    const blob =
        await canvasToBlob(
            frame
        );


    const formData =
        new FormData();


    formData.append(

        "image",

        blob,

        "attendance.jpg"

    );


    formData.append(

        "mode",

        mode

    );


    try {

        const response =
            await fetch(

                "/recognize",

                {

                    method:
                        "POST",

                    body:
                        formData

                }

            );


        const data =
            await response.json();


        console.log(
            data
        );


        // =====================================================
        // SUCCESS
        // =====================================================

        if (
            data.success
        ) {

            recognizedId.innerText =
                data.employee_id;


            recognizedName.innerText =
                data.employee_name;


            recognizedScore.innerText =
                data.score;


            attendanceStatus.innerText =
                data.message;


            attendanceStatus.style.background =
                "#d4edda";


            alert(

                data.message

                + "\n\nEmployee: "
                + data.employee_name

                + "\nEmployee ID: "
                + data.employee_id

                + "\nScore: "
                + data.score

            );

        }

        // =====================================================
        // FAILED
        // =====================================================

        else {

            attendanceStatus.innerText =
                data.message;


            attendanceStatus.style.background =
                "#f8d7da";


            if (
                data.score
            ) {

                recognizedScore.innerText =
                    data.score;

            }


            alert(
                data.message
            );

        }


    }

    catch (error) {

        console.error(
            error
        );


        attendanceStatus.innerText =
            "Server error";


        alert(
            "Unable to connect to the Flask server."
        );

    }

}


// ============================================================
// START
// ============================================================

startCamera();