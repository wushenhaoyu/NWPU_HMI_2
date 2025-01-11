<template>
  <div >
    <el-container>
  <el-main style="background-color: #e9eef3;border-radius: 1.5vw;">
    <div style="display: flex;">    
      <div style="width: 70%;">
        <div style="height: 100%;width: 100%;">
          <div style="text-align: center;font-size: larger;font-weight: 900;color: black;background-color: #e9eef3;">可视化</div>
            <div v-if="isShowImg">            
              <img   :src="imgurl" alt="electron-vue" >
            </div>
            <div v-else>
              <img  style="height: 100%;width: 95%;" src="~@/assets/bg.png" alt="electron-vue" >
            </div>

        </div>      


      </div>
      <div style="width: 30%;height: 90vh;"><div style="height: 4vh;"></div><div style="height: 86vh;background-color: #ffffff;border-radius: 1.5vw;">
        <div style="text-align: center;font-weight: 900;line-height: 10vh;font-size: 25px;">控制面板</div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-select v-model="value" placeholder="模式选择" @change="change">
            <el-option
              v-for="item in options"
              :key="item.value"
              :label="item.label"
              :value="item.value">
            </el-option>
          </el-select>
        </div>

        <div v-if="value==1">
          <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;" @click="control" >
          <div>原图 显示:</div>
          <el-switch
          v-model="isShowImg"
          active-text="开启"
          inactive-text="关闭"
          >
        </el-switch>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"  @click="control">
          <div>增强 显示:</div>
          <el-switch
          v-model="isShowImg2"
          active-text="开启"
          inactive-text="关闭"
          >
        </el-switch>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"  @click="control">
          <div>特征1显示:</div>
          <el-switch
          v-model="isShowImg3"
          active-text="开启"
          inactive-text="关闭"
          >
        </el-switch>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;"  @click="control">
          <div>特征2显示:</div>
          <el-switch
          v-model="isShowImg4"
          active-text="开启"
          inactive-text="关闭"
          >
        </el-switch>
        </div>
        </div>
      

        <div style="height: 20vh;"></div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openYOLO">开始识别</el-button>
        </div>
        <div style="font-weight: 900;margin-top: 2vh;display: flex;justify-content: space-evenly;">
          <el-button type="primary" style="width: 80%;font-weight: 600;" @click="openCamera">开启摄像头</el-button>
        </div>
      </div> </div>
    </div>
  </el-main>
</el-container>
  </div>
</template>

<script>
  import SystemInformation from './LandingPage/SystemInformation'

  export default {
    name: 'landing-page',
    components: { SystemInformation },
    data () {
      return {
        //
        isShowImg:false,

        imgurl_:'http://localhost:8000/video_stream_ori/',
        timestamp: Date.now(),
        options: [{
          value: 1,
          label: 'easy'
        }, {
          value: 2,
          label: 'medium'
        }, {
          value: 3,
          label: 'difficult'
        }],
        value: 1
      }
    },
    mounted () {
      //
    },
    computed: {
      //
      imgurl(){
        return `${this.imgurl_}?t=${this.timestamp}`;
      },
      img1url(){
        return `${this.img1url_}?t=${this.timestamp}`;
      },
      img2url(){
        return `${this.img2url_}?t=${this.timestamp}`;
      },
      img3url(){
        return `${this.img3url_}?t=${this.timestamp}`;
      }
    },
    watch: {
      //
    },
    methods: {
      change(){
        let data = {'weight':this.value }
        this.$http.post('http://127.0.0.1:8000/weight',data)
        .then(response => {
          console.log(response.data);
        })
        .catch(error => {
          console.error(error);
        });
      },
      open (link) {
        this.$electron.shell.openExternal(link)
      },
      control(){
        let data = {
          "origin": this.isShowImg1,
          "feature_one": this.isShowImg3,
          "feature_two": this.isShowImg4,
          "output": this.isShowImg2
        }
        this.$http.post('http://127.0.0.1:8000/control_queues',data,{ headers: { 'Content-Type': 'application/json' } })
        .then(response => {
          console.log(response.data);
        })
        .catch(error => {
          console.error(error);
        });
      },
      openCamera(){
        this.$http.post('http://127.0.0.1:8000/openCam')
        .then(response => {
          console.log(response.data);
        })
        .catch(error => {
          console.error(error);
        });
      },
      openYOLO(){
        this.$http.post('http://127.0.0.1:8000/openYOLO')
        .then(response => {
          console.log(response.data);
        })
        .catch(error => {
          console.error(error);
        });
      }
    }
  }
</script>

<style>
  @import url('https://fonts.googleapis.com/css?family=Source+Sans+Pro');



  body { font-family: "Helvetica Neue",Helvetica,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","微软雅黑",Arial,sans-serif;}






</style>
