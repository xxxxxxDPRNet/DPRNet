// (function (i) {
//     typeof define == "function" && define.amd ? define(i) : i()
// })
//     (function () {
//         "use strict";

//         const i = '[dc-splitter="splitter"]',
//             l = '[dc-splitter="before"]',
//             c = '[dc-splitter="center"]',
//             o = '[dc-splitter="after"]',
//             r = '[dc-splitter="handle"]';
//         class d {
//             constructor(t) {
//                 this.element = t,
//                     this.splitterContainer = t,
//                     this.splitterBefore = t.querySelector(l),
//                     this.splitterCenter = t.querySelector(c),
//                     this.splitterAfter = t.querySelector(o),
//                     this.splitterHandle = t.querySelector(r),
//                     this.isHandleLocked = !0,
//                     [this.splitterBefore, this.splitterCenter, this.splitterAfter].forEach(n => {
//                         n.style.pointerEvents = "none",
//                             n.style.userSelect = "none"
//                     }), this.attachEventHandlers()
//             }
//             attachEventHandlers() {
//                 this.element.addEventListener("mousemove", this.splitterMouseMove.bind(this)),
//                     this.element.addEventListener("mousedown", this.splitterMouseDown.bind(this)),
//                     this.element.addEventListener("mouseup", this.splitterMouseUp.bind(this)),
//                     this.element.addEventListener("touchmove", this.splitterMouseMove.bind(this),
//                         { passive: !1 }),
//                     this.element.addEventListener("touchstart", this.splitterMouseDown.bind(this),
//                         { passive: !1 }),
//                     this.element.addEventListener("touchend", this.splitterMouseUp.bind(this), { passive: !1 })
//             }
//             splitterMouseDown(t) {
//                 this.isHandleLocked && (this.isHandleLocked = !1),
//                     this.splitterMouseMove(t)
//             }
//             splitterMouseUp(t) {
//                 this.isHandleLocked || (this.isHandleLocked = !0)
//             }
//             splitterMouseMove(t) {
//                 if (t.preventDefault(), this.isHandleLocked) return;
//                 const e = this.splitterContainer.getBoundingClientRect(), n = this.splitterHandle;
//                 let s = ((t.clientX || t.touches[0].clientX) - e.left).toFixed();
//                 // console.log(s);
//                 if (s < 0) {
//                     s = 0;
//                 } else {
//                     if (s > e.width) {
//                         s = e.width;
//                     }
//                     console.log(((1 - s / e.width) * 100).toFixed(2));
//                     this.splitterAfter.style.width = `${((1 - s / e.width) * 100).toFixed(2)}%`
//                     this.splitterHandle.style.left = `${(s / e.width * 100).toFixed(2) - (n.clientWidth / e.width / 2 * 100).toFixed(2)}%`
//                 }
//                 // s < 0 ? s = 0 : s > e.width && (s = e.width),
//                 //     this.splitterAfter.style.width = `${((1 - s / e.width) * 100).toFixed(2)}%`,
//                 //     this.splitterHandle.style.left = `${(s / e.width * 100).toFixed(2) - (n.clientWidth / e.width / 2 * 100).toFixed(2)}%`
//             }
//         } document.addEventListener("DOMContentLoaded", h => {
//             document.querySelectorAll(i).forEach((t, e) => {
//                 new d(t)
//             })
//         })
//     });




const i = '[dc-splitter="splitter"]',
    l = '[dc-splitter="before"]',
    c = '[dc-splitter="center"]',
    o = '[dc-splitter="after"]',
    r = '[dc-splitter="handle"]';


class D {
    constructor(t) {
        this.element = t,
            this.splitterContainer = t,
            this.splitterBefore = t.querySelector(l),
            this.splitterCenter = t.querySelector(c),
            this.splitterAfter = t.querySelector(o),
            this.splitterHandleAll = t.querySelectorAll(r),
            this.isHandleLocked = !0,
            this.teargetDom = null,
            this.leftClip = 30,
            this.rightClip = 80,
            [this.splitterBefore, this.splitterCenter, this.splitterAfter].forEach(n => {
                n.style.pointerEvents = "none",
                    n.style.userSelect = "none"
            }), this.splitterHandleAll.forEach(item => {
                item.addEventListener('mousedown', () => {
                    this.teargetDom = item
                })
                item.addEventListener('mouseup', () => {
                    this.teargetDom = null
                })

            }),
            this.handleMouseMove = this.splitterMouseMove.bind(this)
    }
    attachEventHandlers() {

        this.element.addEventListener("mousedown", this.splitterMouseDown.bind(this))
        this.element.addEventListener("mouseup", this.splitterMouseUp.bind(this))

        this.element.addEventListener("touchstart", this.splitterMouseDown.bind(this),
            { passive: false })
        this.element.addEventListener("touchend", this.splitterMouseUp.bind(this), { passive: false })
    }
    splitterMouseDown(t) {
        console.log("mousedown");
        this.element.addEventListener("mousemove", this.handleMouseMove)
        this.element.addEventListener("touchmove", this.handleMouseMove,
            { passive: false })
    }
    splitterMouseUp(t) {
        console.log("mouseup");
        this.element.removeEventListener("mousemove", this.handleMouseMove)
        this.element.removeEventListener("touchmove", this.handleMouseMove,
            { passive: false })
    }
    splitterMouseMove(t) {
        t.preventDefault()
        const e = this.splitterContainer.getBoundingClientRect()
        let s = ((t.clientX || t.touches[0].clientX) - e.left).toFixed();
        if (s < 0) return s = 0
        if (s > e.width) {
            s = e.width;
        }
        if (this.teargetDom) {
            const LeftRight = this.getteargetDomLeft('left')
            const rightRight = this.getteargetDomLeft('right')
            this.splitterCenter.style.zIndex = LeftRight >= rightRight ? -1 : 1
            const centerClip = (100 - ((1 - s / e.width) * 100).toFixed(2)).toFixed(2)
            if (this.JudgeAround() == 'left') {
                this.splitterCenter.style.clipPath = `polygon(${centerClip}% 0, ${this.rightClip}% 0, ${this.rightClip}% 100%, ${centerClip}% 100%)`
                this.leftClip = centerClip
                if (LeftRight >= rightRight) {
                    // console.log(`${((1 - s / e.width) * 100).toFixed(2)}%`);
                    this.splitterAfter.style.width = `${100 - LeftRight}%`
                } else {
                    this.splitterAfter.style.width = `${100 - rightRight}%`
                }
            } else {
                this.rightClip = centerClip
                this.splitterCenter.style.clipPath = `polygon(${this.leftClip}% 0, ${centerClip}% 0, ${centerClip}% 100%, ${this.leftClip}% 100%)`
                this.splitterAfter.style.width = `${((1 - s / e.width) * 100).toFixed(2)}%`
            }
            this.teargetDom.style.left = `${(s / e.width * 100).toFixed(2) - (this.teargetDom.clientWidth / e.width / 2 * 100).toFixed(2)}%`



        }
    }
    JudgeAround() {
        const classList = this.teargetDom.classList
        if ([...classList].includes('is-1')) {
            return 'left'
        } else {
            return 'right'
        }
    }
    getteargetDomLeft(type) {
        let left = 0;
        this.splitterHandleAll.forEach(item => {
            if (item.classList.contains(type === 'left' ? 'is-1' : 'is-4')) {
                left = +item.style.left.replace('%', '')
            }
        })
        return left
    }

}

document.addEventListener("DOMContentLoaded", h => {
    document.querySelectorAll(i).forEach((t, e) => {
        const splitter = new D(t);
        splitter.attachEventHandlers();
    })
})